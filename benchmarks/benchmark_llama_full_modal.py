from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/llama_full_benchmark"
VOLUME_NAME = os.environ.get("LLAMA_BENCH_VOLUME", "llama2-full-model-benchmark")

image = (modal.Image.debian_slim(python_version="3.11") .pip_install("torch==2.8.0", "triton==3.4.0") .pip_install("numpy==2.2.6", "sentencepiece==0.2.1", "transformers==4.56.2") .add_local_file(REPO_ROOT / "models/llama.py", f"{REMOTE_ROOT}/models/llama.py") .add_local_file(REPO_ROOT / "engine/paged_kv_cache.py", f"{REMOTE_ROOT}/engine/paged_kv_cache.py") .add_local_file(REPO_ROOT / "kernels/paged_attention.py", f"{REMOTE_ROOT}/kernels/paged_attention.py"))
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("llama-full-model-attention-benchmark")

@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=1200, retries=0, volumes={f"{REMOTE_ROOT}/llama-2-7b": volume})
def benchmark(batches: list[int], contexts: list[int], new_tokens: int = 16, trials: int = 3) -> dict:
    from collections import defaultdict
    import gc
    import platform
    import sys
    import time
    from types import SimpleNamespace
    from unittest.mock import patch

    import sentencepiece as spm
    import torch
    import triton

    if not batches or not contexts or min(batches + contexts) < 1:
        raise ValueError("Positive batches and context lengths are required")
    if new_tokens < 2 or trials < 1:
        raise ValueError("At least two new tokens and one trial are required")
    if max(contexts) + new_tokens > 4096:
        raise ValueError("The request exceeds this model's RoPE/context limit")

    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer_path = Path("llama-2-7b/tokenizer.model")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(vocab_size=tokenizer.vocab_size())):
        from models import llama
    from engine.paged_kv_cache import PagedKVCache
    from kernels.paged_attention import paged_attention_decode

    llama.device = "cuda"
    checkpoint = Path("llama-2-7b/consolidated.00.pth")
    print(f"Loading local checkpoint ({checkpoint.stat().st_size / 1e9:.2f} GB).", flush=True)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights.pop("rope.freqs", None)
    checkpoint_dtypes = sorted({str(t.dtype) for t in weights.values()})
    print(f"Checkpoint dtypes: {checkpoint_dtypes}; benchmark compute dtype: FP16.", flush=True)
    with torch.device("meta"):
        model = llama.Transformer()
    model.load_state_dict(weights, assign=True)
    model.freqs_cis = llama.precompute_complex_exponential_freqs(llama.head_size, llama.block_size)
    del weights

    model = model.half().to(device="cuda").eval()
    gc.collect()
    if any(p.dtype != torch.float16 for p in model.parameters()):
        raise RuntimeError("Expected all checkpoint parameters to be FP16")
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters across {len(model.layers)} layers in FP16.", flush=True)

    original_attention = llama.Attention.forward
    active= {"backend": "torch", "kernel_calls": 0}

    def attention_adapter(self, x, freqs_cis, start_positions, mask, request_ids, layer_id, kv_cache):
        if active["backend"] == "torch" or x.shape[1] != 1:
            return original_attention(self, x, freqs_cis, start_positions, mask, request_ids, layer_id, kv_cache)
        batch, tokens, channels = x.shape
        query = self.wq(x).view(batch, tokens, llama.n_heads, llama.head_size)
        key = self.wk(x).view(batch, tokens, llama.n_heads, llama.head_size)
        value = self.wv(x).view(batch, tokens, llama.n_heads, llama.head_size)
        query, key = llama.apply_rope(query, key, freqs_cis)
        kv_cache.write(request_ids, start_positions, layer_id, key, value)
        out = paged_attention_decode(query, kv_cache, request_ids, start_positions, layer_id)
        active["kernel_calls"] += 1
        return self.wo(out.contiguous().view(batch, tokens, channels))

    llama.Attention.forward = attention_adapter

    def make_cache(batch, context):
        cache = PagedKVCache.__new__(PagedKVCache)
        cache.tokens_per_page = 16
        cache.num_layers, cache.n_heads, cache.head_dim = llama.n_layers, llama.n_heads, llama.head_size
        cache.page_size = cache.tokens_per_page * cache.num_layers * cache.n_heads * cache.head_dim * 2 * 2
        cache.num_pages = batch * ((context + new_tokens + cache.tokens_per_page - 1) // cache.tokens_per_page) + 2
        cache.total_bytes = cache.num_pages * cache.page_size
        free_bytes, _ = torch.cuda.mem_get_info()
        if cache.total_bytes > free_bytes * 0.6:
            raise RuntimeError("KV pool exceeds 60% of currently free device memory")
        shape = (cache.num_pages, cache.tokens_per_page, cache.num_layers, cache.n_heads, cache.head_dim)
        cache.k_cache = torch.empty(shape, device="cuda", dtype=torch.float16)
        cache.v_cache = torch.empty_like(cache.k_cache)
        cache.page_map = defaultdict(list)
        cache.sequence_lengths = defaultdict(int)
        cache.free= set(range(cache.num_pages))
        return cache

    def make_prompts(batch, length):
        rows = []
        topics = ["computers", "astronomy", "gardening", "music", "mathematics", "history", "cooking", "ocean life"]
        ending = tokenizer.encode("\nA useful lesson from this is", out_type=int)
        for index in range(batch):
            topic = topics[index % len(topics)]
            text = f"This is a passage about {topic}. Learning begins with careful observation. We can ask questions, compare explanations, and test our ideas. Simple examples often help us understand a difficult subject. "
            body = tokenizer.encode(text, out_type=int)
            body_len = length - 1 - len(ending)
            if body_len < 0:
                raise ValueError("Prompt length is too short for the benchmark text")
            tokens = [tokenizer.bos_id()] + (body * ((body_len + len(body) - 1) // len(body)))[:body_len] + ending
            assert len(tokens) == length
            rows.append(tokens)
        return torch.tensor(rows, dtype=torch.long, device="cuda"), rows

    def generate(backend, prompt, cache, forced_tokens=None, warmup=False):
        batch, context = prompt.shape
        active["backend"] = backend
        active["kernel_calls"] = 0

        cache.page_map.clear()
        cache.sequence_lengths.clear()
        cache.free= set(range(cache.num_pages))
        request_ids = [1009 + 17 * i for i in range(batch)]
        for rid in request_ids:
            cache.add_request(rid)
        outputs, traces= [], []
        torch.cuda.synchronize()
        started = time.perf_counter()

        for pos in range(0, context, llama.prefill_chunk_size):
            logits = model(prompt[:, pos:pos + llama.prefill_chunk_size], [pos] * batch, request_ids, cache)
        scores = logits[:, -1, :]
        next_token = scores.argmax(-1, keepdim=True)
        outputs.append(next_token)
        traces.append(scores.detach())
        torch.cuda.synchronize()
        prefill_finished = time.perf_counter()

        for step in range(new_tokens - 1):
            decode_input = next_token if forced_tokens is None else forced_tokens[:, step:step + 1]
            logits = model(decode_input, [context + step] * batch, request_ids, cache)
            scores = logits[:, -1, :]
            next_token = scores.argmax(-1, keepdim=True)
            outputs.append(next_token)
            traces.append(scores.detach())
        torch.cuda.synchronize()
        ended = time.perf_counter()
        generated = torch.cat(outputs, dim=1)
        all_logits = torch.stack(traces, dim=1)
        if not bool(torch.isfinite(all_logits).all()):
            raise RuntimeError(f"Nonfinite logits in {backend}")
        expected_calls = llama.n_layers * (new_tokens - 1) if backend == "triton" else 0
        if active["kernel_calls"] != expected_calls:
            raise AssertionError(f"Expected {expected_calls} kernel calls, got {active['kernel_calls']}")
        result = {
            "backend": backend, "warmup": warmup,
            "prefill_s": prefill_finished - started,
            "decode_s": ended - prefill_finished,
            "total_s": ended - started,
            "generated_token_ids": generated.cpu().tolist(),
            "triton_kernel_calls": active["kernel_calls"],
        }
        for rid in request_ids:
            cache.free_request(rid)
        return result, generated, all_logits

    props = torch.cuda.get_device_properties(0)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"gpu": props.name, "gpu_memory_bytes": props.total_memory, "python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda},
        "settings": {"batches": batches, "contexts": contexts, "new_tokens_per_request": new_tokens, "decode_forwards": new_tokens - 1, "trials": trials, "dtype": "float16", "prefill_chunk_size": llama.prefill_chunk_size, "layers": llama.n_layers, "heads": llama.n_heads, "head_dim": llama.head_size, "selection": "greedy", "eos_stopping": False},
        "scope": {"total": "full model prefill + first greedy token + 15 full model decode forwards and greedy selections for default 16 outputs", "decode": "15 full Transformer forwards including actual cache writes, eager reads or Triton metadata setup, all attention/FFN/norm/output layers", "excluded": "upload, model loading, tokenization, cache allocation, JIT warmup, result detokenization, request scheduler and frontend", "integration": "benchmark-local Attention.forward adapter; production source unchanged", "cache": "workload-sized pool, original layout and production allocation/write/read/free methods", "timing": "GPU-synchronized wall time, one untimed complete generation per backend to compile/warm each workload; three measured alternating-order trials", "correctness": "full-vocabulary logits compared on identical histories using teacher-forced baseline output tokens; timed generation is unforced and output equality is reported"},
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ["models/llama.py", "engine/paged_kv_cache.py", "kernels/paged_attention.py", "llama-2-7b/params.json", "llama-2-7b/tokenizer.model"]},
        "checkpoint": {"name": checkpoint.name, "size_bytes": checkpoint.stat().st_size, "source_parameter_dtypes": checkpoint_dtypes},
        "results": [],
    }
    try:
        for batch in batches:
            for context in contexts:
                print(f"Preparing B={batch}, prompt={context}, new_tokens={new_tokens}.", flush=True)
                cache = make_cache(batch, context)
                prompt, prompt_ids = make_prompts(batch, context)

                eager_warm, reference_tokens, reference_logits = generate("torch", prompt, cache, warmup=True)
                triton_warm, candidate_tokens, candidate_logits = generate("triton", prompt, cache, forced_tokens=reference_tokens, warmup=True)
                diff = (candidate_logits.float() - reference_logits.float()).abs()

                rmse = diff.square().mean().sqrt().item()
                ref_probs = reference_logits.float().softmax(-1)
                candidate_log_probs = candidate_logits.float().log_softmax(-1)
                kl = (ref_probs * (reference_logits.float().log_softmax(-1) - candidate_log_probs)).sum(-1)
                validation = {
                    "max_abs_logit_error": diff.max().item(), "logit_rmse": rmse,
                    "mean_kl_divergence": kl.mean().item(), "max_kl_divergence": kl.max().item(),
                    "top1_agreement": (candidate_logits.argmax(-1) == reference_logits.argmax(-1)).float().mean().item(),
                    "logits_close_atol_0_2_rtol_0_02": torch.allclose(candidate_logits, reference_logits, atol=0.2, rtol=0.02),
                }
                if not validation["logits_close_atol_0_2_rtol_0_02"] or validation["max_kl_divergence"] > 0.002:
                    raise AssertionError(f"Full-model output validation failed: {validation}")
                print(f"Warmup/validation passed: {validation}", flush=True)
                del eager_warm, triton_warm, reference_tokens, candidate_tokens, reference_logits, candidate_logits, diff, ref_probs, candidate_log_probs, kl
                torch.cuda.reset_peak_memory_stats()
                timings = {"torch": [], "triton": []}
                for trial in range(trials):
                    order = ("torch", "triton") if trial % 2 == 0 else ("triton", "torch")
                    for backend in order:
                        item, generated, logits = generate(backend, prompt, cache)
                        item["trial"] = trial
                        timings[backend].append(item)
                        print(f"B={batch} P={context} trial={trial + 1} {backend}: total={item['total_s']:.4f}s decode={item['decode_s']:.4f}s", flush=True)
                        del generated, logits
                summary = {}
                for backend, samples in timings.items():
                    total = statistics.median(item["total_s"] for item in samples)
                    decode = statistics.median(item["decode_s"] for item in samples)
                    summary[backend] = {"median_total_s": total, "median_prefill_s": statistics.median(item["prefill_s"] for item in samples), "median_decode_s": decode, "total_output_tokens_per_s": batch * new_tokens / total, "decode_output_tokens_per_s": batch * (new_tokens - 1) / decode, "decode_ms_per_step": decode * 1000 / (new_tokens - 1)}
                equal = all(timings['torch'][i]['generated_token_ids'] == timings['triton'][i]['generated_token_ids'] for i in range(trials))
                row = {"batch": batch, "prompt_tokens": context, "new_tokens_per_request": new_tokens, "kv_pool_bytes": cache.total_bytes, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "validation": validation, "generated_tokens_equal_all_trials": equal, "prompt_token_ids": prompt_ids, "timings": timings, "summary": summary, "total_speedup": summary['torch']['median_total_s'] / summary['triton']['median_total_s'], "decode_speedup": summary['torch']['median_decode_s'] / summary['triton']['median_decode_s'], "generated_text": {backend: [tokenizer.decode(tokens) for tokens in timings[backend][0]['generated_token_ids']] for backend in timings}}
                report['results'].append(row)
                print(f"RESULT B={batch} P={context}: total {row['total_speedup']:.3f}x; decode {row['decode_speedup']:.3f}x; identical generated tokens={equal}", flush=True)
                del cache, prompt
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        llama.Attention.forward = original_attention
    return report

@app.local_entrypoint()
def main(batches: str = "1,8", contexts: str = "128,2048", new_tokens: int = 16, trials: int = 3, output: str = "benchmarks/results/raw/llama_full_h100_results.json"):
    batch_list = list(dict.fromkeys(int(value.strip()) for value in batches.split(',')))
    context_list = list(dict.fromkeys(int(value.strip()) for value in contexts.split(',')))
    report = benchmark.remote(batch_list, context_list, new_tokens, trials)
    report['source_sha256']['benchmark_llama_full_modal.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Saved results to {destination.resolve()}")

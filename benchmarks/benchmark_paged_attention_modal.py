from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_SOURCES = "/root/paged_attention_benchmark_sources"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "triton==3.4.0")
    .add_local_file(REPO_ROOT / "kernels/paged_attention.py", f"{REMOTE_SOURCES}/paged_attention.py")
    .add_local_file(REPO_ROOT / "engine/paged_kv_cache.py", f"{REMOTE_SOURCES}/paged_kv_cache.py")
)
app = modal.App("paged-attention-benchmark")

@app.function(image=image, gpu="H100", timeout=900, retries=0)
def benchmark(batches: list[int], contexts: list[int], dtype: str = "float16", warmup_ms: int = 25, rep_ms: int = 100, trials: int = 3, seed: int = 123) -> dict:
    import platform
    import sys
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as F
    import triton
    import triton.testing

    sys.path.insert(0, REMOTE_SOURCES)
    from paged_attention import _paged_attention_decode_kernel, paged_attention_decode
    from paged_kv_cache import PagedKVCache

    if not batches or not contexts or any(n < 1 for n in batches + contexts):
        raise ValueError("Batches and context lengths must be positive and nonempty")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dtype must be float16 or bfloat16")
    if min(warmup_ms, rep_ms, trials) < 1:
        raise ValueError("Warmup, repetition duration, and trial count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the Modal GPU")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    data_dtype = getattr(torch, dtype)
    page_size, layers, heads, dim = 16, 32, 32, 128
    atol, rtol = (3e-3, 1e-2) if dtype == "float16" else (2e-2, 5e-2)

    def make_mask(lengths):
        return (torch.arange(max(lengths), device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None])[:, None, None, :]

    def make_case(lengths, layer, case_seed):
        torch.manual_seed(case_seed)
        counts = [(length + page_size - 1) // page_size for length in lengths]
        num_pages= sum(counts) + 2
        shape = (num_pages, page_size, layers, heads, dim)

        cache = PagedKVCache.__new__(PagedKVCache)
        cache.tokens_per_page = page_size
        cache.num_layers, cache.n_heads, cache.head_dim = layers, heads, dim
        cache.k_cache = torch.full(shape, float("nan"), device="cuda", dtype=data_dtype)
        cache.v_cache = torch.full_like(cache.k_cache, float("nan"))
        cache.page_map = {}
        request_ids = [1009 + 17 * i for i in reversed(range(len(lengths)))]
        physical_pages = torch.randperm(num_pages).tolist()
        q = torch.randn((len(lengths), heads, dim), device="cuda", dtype=data_dtype)

        logical_shape = (len(lengths), max(lengths), heads, dim)
        logical_k = torch.randn(logical_shape, device="cuda", dtype=data_dtype)
        logical_v = torch.randn_like(logical_k)
        mask = make_mask(lengths)
        valid = mask[:, 0, 0, :]
        invalid = (~valid)[:, :, None, None]
        logical_k.masked_fill_(invalid, 0)
        logical_v.masked_fill_(invalid, 0)
        cursor = 0
        rows = []
        for b, (rid, length, count) in enumerate(zip(request_ids, lengths, counts)):
            pages = physical_pages[cursor:cursor + count]
            cursor += count
            cache.page_map[rid] = pages
            rows.append(pages)
            for logical_page, physical_page in enumerate(pages):
                start = logical_page * page_size
                size = min(page_size, length - start)
                cache.k_cache[physical_page, :size, layer] = logical_k[b, start:start + size]
                cache.v_cache[physical_page, :size, layer] = logical_v[b, start:start + size]

        width=max(counts)
        table = torch.tensor([row + [-1] * (width - len(row)) for row in rows], device="cuda", dtype = torch.int32)
        lens = torch.tensor(lengths, device="cuda", dtype = torch.int32)
        scores = (q.float().unsqueeze(2) @ logical_k.float().permute(0, 2, 3, 1)) * dim**-0.5
        weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        expected = (weights @ logical_v.float().transpose(1, 2)).squeeze(2)
        return SimpleNamespace(cache=cache, ids=request_ids, lengths=lengths, layer=layer, q=q, query=q.unsqueeze(1), starts=[length - 1 for length in lengths], table=table, lens=lens, mask=mask, out=torch.empty_like(q), expected=expected)

    def direct_kernel(case):
        _paged_attention_decode_kernel[(len(case.ids), heads)](case.q, case.cache.k_cache, case.cache.v_cache, case.table, case.lens, case.out, case.layer, *case.cache.k_cache.stride(), NUM_HEADS=heads, HEAD_DIM=dim, PAGE_SIZE=page_size, TABLE_WIDTH=case.table.shape[1], SCALE=dim**-0.5, num_warps=4)
        return case.out

    def wrapper(case):
        return paged_attention_decode(case.query, case.cache, case.ids, case.starts, case.layer).squeeze(1)

    def gathered_attention(case, use_sdpa=False, rebuild_mask=False):
        mask = make_mask(case.lengths) if rebuild_mask else case.mask
        k, v = case.cache.read(case.ids, case.lengths, case.layer)
        q, k, v = case.q.unsqueeze(2), k.transpose(1, 2), v.transpose(1, 2)
        if use_sdpa:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) * dim**-0.5
            weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            output = weights @ v
        return output.squeeze(2)

    def validate(case):
        errors = {}
        functions = {
            "triton_kernel": lambda: direct_kernel(case),
            "triton_wrapper": lambda: wrapper(case),
            "gather_eager": lambda: gathered_attention(case),
            "gather_sdpa": lambda: gathered_attention(case, use_sdpa=True),
        }
        for name, fn in functions.items():
            actual = fn().float()
            torch.testing.assert_close(actual, case.expected, atol=atol, rtol=rtol)
            errors[name] = (actual - case.expected).abs().max().item()
        torch.cuda.synchronize()
        return errors

    def wall_timing(fn):
        iterations = 10
        for _ in range(3):
            fn()
        samples = []
        for _ in range(trials):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000 / iterations)
        return {"median_ms": statistics.median(samples), "trial_ms": samples}

    properties = torch.cuda.get_device_properties(0)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": properties.name, "compute_capability": list(torch.cuda.get_device_capability()),
            "gpu_memory_bytes": properties.total_memory, "python": platform.python_version(),
            "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
        },
        "source_sha256": {
            name: hashlib.sha256(Path(REMOTE_SOURCES, name).read_bytes()).hexdigest()
            for name in ("paged_attention.py", "paged_kv_cache.py")
        },
        "settings": {
            "dtype": dtype, "page_size": page_size, "layers": layers, "heads": heads,
            "head_dim": dim, "batches": batches, "contexts": contexts,
            "seed": seed, "warmup_ms": warmup_ms, "rep_ms": rep_ms, "trials": trials,
            "atol": atol, "rtol": rtol, "num_warps": 4,
        },
        "scope": {
            "included": "one layer of T=1 attention over synthetic, shuffled KV pages",
            "excluded": "model weights, projections, RoPE, cache writes, scheduler, sampling, JIT compilation",
            "gpu_event_timing": "prebuilt metadata; preallocated Triton output; actual gather/read and temporary allocations in baselines; do_bench cache clearing; Python submission gaps may be included",
            "wall_timing": "synchronized repeated calls with metadata/output preparation; no explicit cache flush; mask rebuilt for baselines; actual Triton wrapper",
            "sdpa_backend": "selected automatically by PyTorch, not forced to FlashAttention",
            "memory_traffic": "not measured; latency results do not establish fewer device-memory accesses",
        },
        "correctness": [],
        "results": [],
    }

    checks = [([1], 0), ([16], 15), ([17], 31), ([1, 16, 17, 33], 15), ([65, 7, 2, 32], 0)]
    for index, (lengths, layer) in enumerate(checks):
        case = make_case(lengths, layer, seed + index)
        report["correctness"].append({ "lengths": lengths, "layer": layer, "max_abs_error": validate(case), })
        del case
    print("Correctness checks passed; starting latency measurements.", flush=True)
    print("B  context  lengths   eager_us  SDPA_us  Triton_us  eager/Triton  SDPA/Triton", flush=True)

    for batch in batches:
        for context in contexts:
            profiles = ["uniform"] if batch == 1 else ["uniform", "ragged"]
            for profile in profiles:
                lengths = [context] * batch if profile == "uniform" else [
                    max(1, context - (i * 53) % max(1, context // 2)) for i in range(batch)
                ]
                row = {"batch": batch, "max_context": context, "profile": profile, "lengths": lengths}
                torch.cuda.empty_cache()
                free_bytes, _ = torch.cuda.mem_get_info()
                pages = sum((length + page_size - 1) // page_size for length in lengths) + 2
                cache_bytes = pages * page_size * layers * heads * dim * 2 * 2
                if cache_bytes > free_bytes * 0.6:
                    row.update(status="skipped", reason="KV buffers would exceed 60% of free GPU memory")
                    report["results"].append(row)
                    print(row, flush=True)
                    continue

                case = make_case(lengths, 15, seed + 100 + len(report["results"]))
                row["max_abs_error"] = validate(case)
                functions = {
                    "triton_kernel": lambda: direct_kernel(case),
                    "gather_eager": lambda: gathered_attention(case),
                    "gather_sdpa": lambda: gathered_attention(case, use_sdpa=True),
                }
                samples = {name: [] for name in functions}
                names = list(functions)
                for trial in range(trials):
                    for name in (names if trial % 2 == 0 else names[::-1]):
                        p20, median, p80 = triton.testing.do_bench(functions[name], warmup=warmup_ms, rep=rep_ms, quantiles=[0.2, 0.5, 0.8])
                        samples[name].append({"p20_ms": p20, "median_ms": median, "p80_ms": p80})
                medians = {
                    name: statistics.median(sample["median_ms"] for sample in values)
                    for name, values in samples.items()
                }
                row.update(status="passed", gpu_event_trials=samples, median_gpu_event_ms=medians, eager_over_triton=medians["gather_eager"] / medians["triton_kernel"], sdpa_over_triton=medians["gather_sdpa"] / medians["triton_kernel"], wall_with_metadata={ "triton_wrapper": wall_timing(lambda: wrapper(case)), "gather_eager": wall_timing(lambda: gathered_attention(case, rebuild_mask=True)), "gather_sdpa": wall_timing(lambda: gathered_attention(case, use_sdpa=True, rebuild_mask=True)), })
                report["results"].append(row)
                print(f"{batch:2} {context:8} {profile:8} " f"{medians['gather_eager'] * 1000:9.2f} {medians['gather_sdpa'] * 1000:8.2f} " f"{medians['triton_kernel'] * 1000:9.2f} {row['eager_over_triton']:13.2f} " f"{row['sdpa_over_triton']:12.2f}", flush=True)
                del functions, case
    return report

@app.local_entrypoint()
def main(batches: str = "1,4,8", contexts: str = "128,512,2048", dtype: str = "float16", output: str = "benchmarks/results/raw/paged_attention_h100_results.json", warmup_ms: int = 25, rep_ms: int = 100, trials: int = 3, seed: int = 123):
    def parse_sizes(value):
        sizes = list(dict.fromkeys(int(item.strip()) for item in value.split(",")))
        if not sizes or any(size < 1 for size in sizes):
            raise ValueError("Sizes must be positive comma-separated integers")
        return sizes

    batch_sizes, context_sizes = parse_sizes(batches), parse_sizes(contexts)
    if dtype not in ("float16", "bfloat16") or min(warmup_ms, rep_ms, trials) < 1:
        raise ValueError("Use float16/bfloat16 and positive timing parameters")
    report = benchmark.remote(batch_sizes, context_sizes, dtype, warmup_ms, rep_ms, trials, seed)
    report["source_sha256"]["benchmark_paged_attention_modal.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved results to {destination.resolve()}")

# Llama 2 7B Inference Engine

Llama 2 7B built from scratch in PyTorch, with a serving scheduler, continuous batching, and a paged KV cache.

## Progress

- [x] Llama 2 7B: RMSNorm, MHA, SwiGLU, and RoPE
- [x] Chunked prefill and batched decoding across different context lengths
- [x] Per-request KV page allocation and reuse
- [x] Continuous batching and request scheduling
- [x] Triton paged attention kernel for decoding
- [ ] Mixed prefill/decode batching
- [ ] Quantization
- [ ] Speculative decoding

## Code

- [models/llama.py](models/llama.py): model and standalone generation.
- [engine/inference_engine.py](engine/inference_engine.py): scheduling, prefill, and batched decoding.
- [engine/paged_kv_cache.py](engine/paged_kv_cache.py): KV page allocation and storage.
- [kernels/paged_attention.py](kernels/paged_attention.py): decode attention with direct KV-page access and online softmax.
- [models/llama_no_kv_cache.py](models/llama_no_kv_cache.py): uncached baseline.

Run `python -m models.llama` from the repo root. Requires PyTorch, Transformers, SentencePiece, the Llama 2 7B checkpoint, and access to the `meta-llama/Llama-2-7b-hf` tokenizer.

The Triton kernel requires CUDA and Triton; integration into the engine is pending.

## Results

On **1×H100**:

- Pre-allocated KV caching improved throughput **7.74× at 2K context** versus no cache.
- Triton paged attention achieved **8.25× faster full-model decode** (4.011 s → 0.486 s) versus the PyTorch attention baseline, and **1.13× faster total generation** including prefill.

Triton results used a benchmark adapter: FP16, batch 8, 2,048 prompt tokens and 16 new tokens per request, median of three trials. Decode covers 15 forwards after prefill; loading and warmup are excluded. Generated tokens matched across both paths.

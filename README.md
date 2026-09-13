# Llama 2 7B Inference Engine

A from scratch **Llama 2 7B inference engine** in PyTorch, building toward an **inference serving engine** with request scheduling and continuous batching.

## Progress

- [x] Llama 2 7B inference
- [x] KV cache + chunked prefill
- [x] Paged KV allocation
- [ ] Continuous batching + serving scheduler
- [ ] Paged attention kernel
- [ ] Quantization
- [ ] Speculative decoding

## Code

- [llama.py](llama.py): model, prefill, decoding, and timing.
- [paged_kv_cache.py](paged_kv_cache.py): per-request KV page allocation and storage.
- [llama_no_kv_cache.py](llama_no_kv_cache.py): uncached baseline.

Run `python llama.py` from the repo root with PyTorch, Transformers, and SentencePiece installed. Requires access to llama-2-7b and access to the `meta-llama/Llama-2-7b-hf` tokenizer.

## Results

On **1×H100**, a pre-allocated KV cache with position-indexed writes improved throughput **7.74× at 2K context** versus no cache.

More results coming soon...
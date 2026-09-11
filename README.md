# Llama 2 7B Inference Engine

A from-scratch **Llama 2 7B inference engine** in PyTorch, building toward an **inference serving engine** with request scheduling and continuous batching. Currently runs single-request inference on Apple Silicon (MPS).

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

Run `python llama.py` from the repo root with PyTorch, Transformers, and SentencePiece installed. Requires `llama-2-7b/{params.json,consolidated.00.pth}` and access to the `meta-llama/Llama-2-7b-hf` tokenizer.

## Learning projects

- [bigram_model.py](learning/bigram_model.py): Andrej Karpathy's character-level bigram model on Tiny Shakespeare.
- [gpt.py](learning/gpt.py): character-level transformer trained on Tiny Shakespeare.
- [pytorch_learning/](learning/pytorch_learning/): pytorch practice + learning
- [cuda_practice/](learning/cuda_practice/): cuda kernel practice + learning
- [for_colab/](learning/for_colab/): bigram/self-attention walkthrough, GPT training, and Llama 2 inference notebooks.

Run training scripts from `learning/` with Tiny Shakespeare saved as `input.txt`.

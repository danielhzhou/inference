# Llama 2 7B generation

H100 80GB, fp16. 16 new tokens per request, greedy decoding, EOS stopping disabled. Median of 3 trials after warmup, alternating which version ran first.

Decode time covers 15 passes through the full model. Prefill produces the first token.

| Batch | Prompt tokens | PyTorch decode (s) | Triton decode (s) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.368 | 0.342 | 1.07x |
| 1 | 2,048 | 0.786 | 0.337 | 2.34x |
| 8 | 128 | 0.827 | 0.405 | 2.04x |
| 8 | 2,048 | 4.011 | 0.486 | 8.25x |

At batch 8 / 2K, total generation took 26.030 s with PyTorch and 23.110 s with Triton: 1.13x faster. Prefill took about 22 seconds either way, so it dominated this short run.

Only decode attention changed, through a benchmark adapter. Both versions used 32-token prefill chunks and a KV pool sized for the workload. Timing includes cache writes and metadata setup; loading, tokenization, cache allocation, and warmup are excluded.

Generated tokens matched in every trial. Comparing logits on the same histories gave a maximum difference of 0.046875. The batch-1/128-token gain was small and varied across trials. The scheduler wasn't tested.

[Raw data](raw/llama_full_h100_results.json) / [Script](../benchmark_llama_full_modal.py)

# Paged attention

Single-layer decode on an H100 80GB, fp16. Median of 3 trials with GPU synchronization. Times include cache access and metadata setup.

2,048 cached tokens per request:

| Batch | Read + PyTorch (ms) | Read + SDPA (ms) | Triton (ms) | Speedup vs PyTorch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.082 | 1.014 | 0.286 | 3.79x |
| 4 | 4.138 | 3.852 | 0.309 | 13.38x |
| 8 | 8.432 | 7.735 | 0.391 | 21.58x |

Both PyTorch paths gather K/V into dense tensors through `cache.read()`. Triton reads directly from the pages. The SDPA column includes that gathering cost too.

Used random K/V, shuffled pages, 32 heads, 128 dimensions per head, and 16 tokens per page. Tested batches 1/4/8 at contexts 128/512/2,048, including unequal lengths. All 15 workloads and 5 boundary/layout checks passed against FP32 attention. Largest Triton error was 0.000965.

The 21.58x result is for one attention call. Full-model decode reached 8.25x in the [generation test](llama_full_h100_results.md).

[Raw data](raw/paged_attention_h100_results.json) / [Script](../benchmark_paged_attention_modal.py)

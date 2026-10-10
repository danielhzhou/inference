# KV cache vs no cache

Original manual result on one H100: 7.74x higher throughput with a preallocated KV cache, reported at 2K context.

The cached version prefills the prompt once, then processes one new token per step. It stores K/V in fixed tensors using position-indexed writes. The uncached version runs the full growing sequence through the model each step.

The timing loops measure generation with `time.perf_counter()` and device synchronization, then print elapsed seconds and `max_tokens / elapsed`. Model loading and final detokenization sit outside the timer.


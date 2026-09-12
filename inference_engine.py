from collections import deque
from typing import List

from llama import Transformer, device, head_size, max_batch_size, n_heads, n_layers
from paged_kv_cache import PagedKVCache
from request import InferenceRequest

class InferenceEngine:
    def __init__(self, weights):
        self.model = Transformer()
        self.model.load_state_dict(weights)
        self.model = self.model.to(device)
        model_dtype = next(self.model.parameters()).dtype

        self.kv_cache = PagedKVCache(n_layers, n_heads, head_size, device, model_dtype)
        self.waiting_queue = deque() # list of requests
        self.processing_queue = [None] * max_batch_size

    def add_request(self, request: InferenceRequest) -> None:
        pass

    def finish_request(self, request: InferenceRequest) -> None:
        pass

    def prefill(self, request: InferenceRequest, chunk_size: int) -> None:
        """Process one prompt chunk; sample the first output token
        if prefill finishes."""

    def decode(self, requests: List[InferenceRequest]) -> None:
        """Process one pending token per request in a batch,
        then sample one new token per request."""

    def step(self) -> None:
        """Admit waiting requests that fit.
        Run a decode batch and a prefill chunk.
        Finish requests that hit EOS or their token limit."""

    def run_generation_loop(self) -> None:
        pass

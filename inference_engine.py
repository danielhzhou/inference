from collections import deque
from dataclasses import dataclass
from enum import Enum

from typing import List

from transformers import AutoTokenizer
from llama import Transformer, device, head_size, max_batch_size, n_heads, n_layers
from paged_kv_cache import PagedKVCache
from request import InferenceRequest

class FinishReason(Enum):
    EOS = 1
    LENGTH = 2
    ERROR = 3

@dataclass
class FinishedRequest:
    finish_reason: FinishReason
    output: str
    request: InferenceRequest

class InferenceEngine:
    def __init__(self, weights):
        self.model = Transformer()
        self.model.load_state_dict(weights)
        self.model = self.model.to(device)
        model_dtype = next(self.model.parameters()).dtype

        self.kv_cache = PagedKVCache(n_layers, n_heads, head_size, device, model_dtype)
        self.waiting_queue = deque() # list of requests
        self.ids_to_process = set() # dedupe by id
        self.processing_queue = [None] * max_batch_size
        self.curr_running_requests = 0

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def add_request(self, request: InferenceRequest) -> None:
        if request.id in self.ids_to_process:
            raise ValueError("id already exists")
        self.waiting_queue.append(request)
        self.ids_to_process.add(request.id)
        
    def finish_request(self, request: InferenceRequest, finish_reason: FinishReason) -> FinishedRequest:
        all_tokens = request.prompt + request.generated_tokens
        output = self.tokenizer.decode(all_tokens)

        self.ids_to_process.remove(request.id)
        self.kv_cache.free_request(request.id)

        return FinishedRequest(finish_reason, output, request.copy())

    def prefill(self, request: InferenceRequest, chunk_size: int) -> None:
        """Process one prompt chunk; sample the first output token
        if prefill finishes."""
        pass

    def decode(self, requests: List[InferenceRequest]) -> None:
        """Process one pending token per request in a batch,
        then sample one new token per request."""
        pass

    def step(self) -> None:
        """Admit waiting requests that fit.
        Run a decode batch and a prefill chunk.
        Finish requests that hit EOS or their token limit."""
        pass

    def run_generation_loop(self) -> None:
        while True:
            self.step()

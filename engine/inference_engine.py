from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from engine.paged_kv_cache import PagedKVCache
from engine.request import InferenceRequest
from models.llama import Transformer, device, head_size, max_batch_size, n_heads, n_layers

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
        self.prefill_chunk_size = 32

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

        return FinishedRequest(finish_reason, output, request)

    def prefill(self, request: InferenceRequest, chunk_size: int) -> None:
        """Process one prompt chunk; sample the first output token
        if prefill finishes."""
        # batched prefill
        tokens_processed = request.num_cached
        if tokens_processed < len(request.prompt):
            remaining_tokens = len(request.prompt) - tokens_processed
            chunk_size = min(chunk_size, remaining_tokens)
            chunk = request.prompt[tokens_processed:tokens_processed + chunk_size]
            chunk = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)

            logits = self.model(chunk, tokens_processed, request.id, self.kv_cache)

            tokens_processed += chunk_size
            request.num_cached = tokens_processed

        # prefill finished, sample first output
        if request.num_cached >= len(request.prompt):
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            request.generated_tokens.append(token_id)

    def decode(self, requests: List[InferenceRequest]) -> None:
        """Process one pending token per request in a batch,
        then sample one new token per request."""
        pass

    # TODO: utilize padding or metadata passed to the transformer in order to 
    # process mixed prefill / decode instead of processing independently
    def step(self) -> None:
        """Admit waiting requests that fit.
        Run a decode batch and a prefill chunk.
        Finish requests that hit EOS or their token limit."""

        free_slots = deque()
        for i in range(len(self.processing_queue)):
            if not self.processing_queue[i]:
                free_slots.append(i)

        # admit waiting requests
        while self.curr_running_requests < max_batch_size and self.waiting_queue:
            new_request = self.waiting_queue.popleft()
            if len(new_request.prompt) >= new_request.max_tokens:
                print("prompt too large")
                self.finish_request(new_request, FinishReason.ERROR)
                continue
            new_index = free_slots.popleft()
            self.kv_cache.add_request(new_request.id)
            self.processing_queue[new_index] = new_request
            self.curr_running_requests += 1

        # anything that needs prefill run prefill
        for request in self.processing_queue:
            if not request:
                continue
            if request.num_cached < len(request.prompt):
                self.prefill(request, self.prefill_chunk_size)

        self._process_finished_requests()

        decode_requests = []
        # run a full decode batch on all processed
        for request in self.processing_queue:
            if not request:
                continue
            if request.num_cached >= len(request.prompt):
                decode_requests.append(request)
            
        if decode_requests:
            self.decode(decode_requests)

        self._process_finished_requests()
        
    def run_generation_loop(self) -> None:
        while True:
            self.step()

    def _process_finished_requests(self) -> None:
        # finish requests that errored, hit EOS, or token limit
        for idx, request in enumerate(self.processing_queue):
            if not request:
                continue
            total_length = len(request.prompt) + len(request.generated_tokens)
            if (request.generated_tokens and request.generated_tokens[-1] == self.tokenizer.eos_token_id):
                self.finish_request(request, FinishReason.EOS)
            elif total_length >= request.max_tokens:
                self.finish_request(request, FinishReason.LENGTH)
            else:
                continue
            self.processing_queue[idx] = None
            self.curr_running_requests -= 1


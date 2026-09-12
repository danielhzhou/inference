from collections import deque

from llama import Transformer, max_batch_size, n_layers, n_heads, head_size, device
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




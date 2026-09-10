from collections import defaultdict
import torch
#TODO finish KV cache impl 
class PagedKVCache:
    def __init__(self, num_layers, n_heads, head_dim, device):
        # llama2 params
        self.page_map = defaultdict(list) # request -> page numbers of pages allocated
        self.total_bytes = 4 * 1024**3 # 4 GiB
        self.num_pages = self.total_bytes // self.page_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.tokens_per_page = 16

        self.page_size = self.tokens_per_page * self.num_layers * self.n_heads * self.head_dim * 2 * 2

        # continuous batching
        # self.sequence_lengths = defaultdict(int)

        self.free = set([i for i in range(self.num_pages)])

        # actual memory block
        self.k_cache = torch.empty(self.num_pages, self.tokens_per_page, self.num_layers, self.n_heads, self.head_dim, dtype=torch.float16, device=device)
        self.v_cache = torch.empty(self.num_pages, self.tokens_per_page, self.num_layers, self.n_heads, self.head_dim, dtype=torch.float16, device=device)

    def add_request(self, request_id):
        # init request
        if request_id in self.page_map:
            raise ValueError("request exists already")

        self.page_map[request_id] = []

    def _palloc(self, request_id):
        # internally allocate a page
        if not self.free:
            raise RuntimeError("oom")

        allocated_page = self.free.pop()

        self.page_map[request_id].append(allocated_page)

        return allocated_page

    def free_request(self, request_id):
        # releases all pages owned by this request
        pages = self.page_map.pop(request_id, [])

        for page in pages:
            self.free.add(page)

    def write(self, request_id, start_pos, layer_idx, key, value):
        # write to the KV cache
        if request_id not in self.page_map:
            raise ValueError("request does not exist")

        B, T, n_head, head_size = key.shape

        if n_head != self.n_heads:
            raise ValueError("mismatch! num heads")
        if head_size != self.head_dim:
            raise ValueError("mismatch! head dim")

        for t in range(T):
            token_pos = start_pos + t

            logical_page = token_pos // self.tokens_per_page
            offset = token_pos % self.tokens_per_page

            while logical_page >= len(self.page_map[request_id]):
                self._palloc(request_id)

            physical_page = self.page_map[request_id][logical_page]

            self.k_cache[physical_page, offset, layer_idx] = key[0, t]
            self.v_cache[physical_page, offset, layer_idx] = value[0, t]

    def read(self, request_id, layer_idx, seq_len):
        # returns (key, value) tuple
        if request_id not in self.page_map:
            raise ValueError("request does not exist")

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        tokens_left = seq_len
        k = []
        v = []

        for physical_page in self.page_map[request_id]:
            if tokens_left <= 0:
                break

            tokens_to_read = min(self.tokens_per_page, tokens_left)

            k.append(self.k_cache[physical_page, :tokens_to_read, layer_idx])
            v.append(self.v_cache[physical_page, :tokens_to_read, layer_idx])

            tokens_left -= tokens_to_read

        if tokens_left > 0:
            raise ValueError("not enough KV tokens allocated")

        key = torch.cat(k, dim=0)
        value = torch.cat(v, dim=0)

        return (key.unsqueeze(0), value.unsqueeze(0))
            
    def get_page_table(self, request_id):
        # return page table for this request
        if request_id not in self.page_map:
            raise ValueError("request does not exist")

        return self.page_map[request_id]

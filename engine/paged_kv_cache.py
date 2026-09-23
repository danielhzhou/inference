from collections import defaultdict
import torch
import torch.nn.functional as F

class PagedKVCache:
    def __init__(self, num_layers, n_heads, head_dim, device, dtype):
        # llama2 params
        self.page_map = defaultdict(list) # request -> page numbers of pages allocated
        self.total_bytes = 4 * 1024**3 # 4 GiB
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.tokens_per_page = 16

        self.page_size = self.tokens_per_page * self.num_layers * self.n_heads * self.head_dim * 2 * 2
        self.num_pages = self.total_bytes // self.page_size

        # continuous batching
        self.sequence_lengths = defaultdict(int)

        self.free = set([i for i in range(self.num_pages)])

        # actual memory block
        self.k_cache = torch.empty(self.num_pages, self.tokens_per_page, self.num_layers, self.n_heads, self.head_dim, dtype=dtype, device=device)
        self.v_cache = torch.empty(self.num_pages, self.tokens_per_page, self.num_layers, self.n_heads, self.head_dim, dtype=dtype, device=device)

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

    def write(self, request_ids, start_positions, layer_idx, key, value):
        if not request_ids or len(request_ids) != len(start_positions):
            raise ValueError("expected matching, nonempty request ids and start positions")
        B, T, n_head, head_size = key.shape

        if n_head != self.n_heads:
            raise ValueError("mismatch! num heads")
        if head_size != self.head_dim:
            raise ValueError("mismatch! head dim")
        # write to the KV cache
        for b, request_id in enumerate(request_ids):
            if request_id not in self.page_map:
                raise ValueError(f"request does not exist at position {b}")

            for t in range(T):
                token_pos = start_positions[b] + t

                logical_page = token_pos // self.tokens_per_page
                offset = token_pos % self.tokens_per_page

                while logical_page >= len(self.page_map[request_id]):
                    self._palloc(request_id)

                physical_page = self.page_map[request_id][logical_page]

                self.k_cache[physical_page, offset, layer_idx] = key[b, t]
                self.v_cache[physical_page, offset, layer_idx] = value[b, t]

    def read(self, request_ids, seq_lens, layer_idx):
        # returns tuple of (keys, vals)
        if not request_ids or len(request_ids) != len(seq_lens):
            raise ValueError("expected matching, nonempty request_ids and seq_lens")
        max_len = max(seq_lens)
        padded_keys = []
        padded_vals = []
        for b, request_id in enumerate(request_ids):

            if request_id not in self.page_map:
                raise ValueError("request does not exist")

            if seq_lens[b] <= 0:
                raise ValueError(f"seq_len at batch idx {b} must be positive")

            tokens_left = seq_lens[b]
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
            val = torch.cat(v, dim=0)

            pad_len = max_len - key.shape[0]
            padded_key = F.pad(key, (0, 0, 0, 0, 0, pad_len))
            padded_keys.append(padded_key)
            padded_val = F.pad(val, (0, 0, 0, 0, 0, pad_len))
            padded_vals.append(padded_val)

        keys = torch.stack(padded_keys, dim=0)
        vals = torch.stack(padded_vals, dim=0)

        return (keys, vals)
            
    def get_page_table(self, request_id):
        # return page table for this request
        if request_id not in self.page_map:
            raise ValueError("request does not exist")

        return self.page_map[request_id]

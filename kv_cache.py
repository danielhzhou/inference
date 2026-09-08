from collections import defaultdict
#TODO finish KV cache impl 
class KVCache:
    def __init__(self):
        self.page_map = defaultdict(list) # request -> page numbers of pages allocated
        self.total_bytes = 4 * 1024**3 # 4 GiB
        self.page_size = 16 * 512 * 1024 # 16 tokens in FP16
        self.num_pages = self.total_bytes // self.page_size

        self.free = set([i for i in range(self.num_pages)])

    def palloc(self, request_id):
        if not self.free:
            raise RuntimeError("oom")

        allocated_page = self.free.pop()

        self.page_map[request_id].append(allocated_page)

        return allocated_page

    
    
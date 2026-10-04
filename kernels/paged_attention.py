import torch
import triton
import triton.language as tl

@triton.jit
def _paged_attention_decode_kernel(Q, K_CACHE, V_CACHE, BLOCK_TABLE, SEQ_LENS, OUT, layer_idx, S_PAGE: tl.constexpr, S_TOKEN: tl.constexpr, S_LAYER: tl.constexpr, S_HEAD: tl.constexpr, S_DIM: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, PAGE_SIZE: tl.constexpr, TABLE_WIDTH: tl.constexpr, SCALE: tl.constexpr):
    batch_idx = tl.program_id(0)
    head_idx= tl.program_id(1)

    dims = tl.arange(0, HEAD_DIM)
    page_offsets = tl.arange(0,PAGE_SIZE)
    query_offsets = (batch_idx * NUM_HEADS + head_idx) * HEAD_DIM + dims
    query = tl.load(Q + query_offsets).to(tl.float32)
    seq_len = tl.load(SEQ_LENS + batch_idx)

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.full((), 0.0, tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    num_pages=tl.cdiv(seq_len, PAGE_SIZE)

    for logical_page in range(num_pages):
        physical_page = tl.load(BLOCK_TABLE + batch_idx * TABLE_WIDTH + logical_page).to(tl.int64)
        token_positions = logical_page * PAGE_SIZE + page_offsets
        valid_tokens = token_positions < seq_len

        cache_offsets = (physical_page * S_PAGE + page_offsets[:, None] * S_TOKEN
                         + layer_idx * S_LAYER + head_idx * S_HEAD + dims[None, :] * S_DIM)
        keys = tl.load(K_CACHE + cache_offsets, mask=valid_tokens[:, None], other=0.0).to(tl.float32)
        values = tl.load(V_CACHE + cache_offsets, mask = valid_tokens[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))
        page_max = tl.max(scores, axis = 0)
        new_max = tl.maximum(running_max, page_max)
        correction = tl.exp(running_max - new_max)
        weights=tl.exp(scores - new_max)

        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        running_sum = running_sum * correction + tl.sum(weights, axis=0)
        running_max = new_max

    output = accumulator / running_sum
    tl.store(OUT + query_offsets, output)


@torch.inference_mode()
def paged_attention_decode(query, kv_cache, request_ids, start_positions, layer_idx):
    assert query.ndim == 4
    B, T, H, D = query.shape
    assert B > 0 and T == 1
    assert len(request_ids) == len(start_positions) == B
    assert all(position >= 0 for position in start_positions)
    assert query.is_cuda, "Requires a supported Triton GPU backend"

    k_cache = kv_cache.k_cache
    v_cache=kv_cache.v_cache
    assert k_cache.ndim == 5
    assert k_cache.shape == v_cache.shape
    assert k_cache.stride() == v_cache.stride()
    assert query.device == k_cache.device == v_cache.device
    assert query.dtype == k_cache.dtype == v_cache.dtype
    assert query.dtype in (torch.float16, torch.bfloat16, torch.float32)

    num_physical_pages, page_size, num_layers, kv_heads, kv_dim = k_cache.shape
    assert H == kv_heads and D == kv_dim
    assert 0 <= layer_idx < num_layers
    assert D > 0 and (D & (D - 1)) == 0
    assert page_size > 0 and (page_size & (page_size - 1)) == 0

    q = query[:, 0].contiguous()
    seq_lens = [position + 1 for position in start_positions]
    page_rows = []
    for request_id, seq_len in zip(request_ids, seq_lens):
        required_pages = (seq_len + page_size - 1) // page_size
        pages = kv_cache.page_map[request_id]
        assert len(pages) >= required_pages
        row=pages[:required_pages]
        assert all(0 <= page < num_physical_pages for page in row)
        page_rows.append(row)

    table_width = max(len(row) for row in page_rows)
    block_table = torch.tensor([row + [-1] * (table_width - len(row)) for row in page_rows], dtype=torch.int32, device=q.device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype = torch.int32, device=q.device)
    out = torch.empty_like(q)

    _paged_attention_decode_kernel[(B, H)](q, k_cache, v_cache, block_table, seq_lens_tensor, out, layer_idx, *k_cache.stride(), NUM_HEADS=H, HEAD_DIM=D, PAGE_SIZE=page_size, TABLE_WIDTH=table_width, SCALE=D**-0.5, num_warps=4)
    return out.unsqueeze(1)

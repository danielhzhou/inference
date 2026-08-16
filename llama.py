import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoTokenizer

device = "mps"

with open("./llama-2-7b/params.json", "r") as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# hyperparams
block_size = 4096 # i think this is typical for Llama 2 7B
n_embd = config["dim"]
n_head = config["n_heads"]
head_size = n_embd // n_head
n_layers = config["n_layers"]
eps = config["norm_eps"]
multiple_of = config["multiple_of"]

max_batch_size = 32
max_seq_len = 2048

def precompute_complex_exponential_freqs(head_size, end, theta = 10000.0):
    """
    end = end index
    theta = scaling factor for frequency computation
    split tensor into groups of 2 to perform rotations
    return in complex64 datatype
    a + ib = r(cos(theta) + isin(theta))
    
    """
    # calc rotation freq, make sure n_head is even
    freqs = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size)) 
    # array of positions
    t_pos = torch.arange(end, device=freqs.device) 
    # pos x freq
    freqs = torch.outer(t_pos, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex num for rotation
    return freqs_cis

# allow pytorch broadcasts
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rope(query, key, freqs_cis):
    q = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    k = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    # allow pytorch to broadcast the tensor
    freqs_cis = reshape_for_broadcast(freqs_cis, q) 
    # perform rotation
    q_out = torch.view_as_real(q * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k * freqs_cis).flatten(3)
    return q_out.type_as(query), k_out.type_as(key)


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight
    

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.wq = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.wv = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.wo = nn.Linear(n_head * head_size, n_embd, bias=False)

        # fixed size kv cache
        self.register_buffer(
            "cache_k",
            torch.zeros(max_batch_size, max_seq_len, n_head, head_size),
            persistent=False,
        )

        self.register_buffer(
            "cache_v",
            torch.zeros(max_batch_size, max_seq_len, n_head, head_size),
            persistent=False,
        )
        
    def forward(self, x, freqs_cis, start_pos):
        B, T, C = x.shape

        q = self.wq(x) # (B, T, 4096)
        k = self.wk(x) 
        v = self.wv(x)

        # split into attention heads
        query = q.view(B, T, n_head, head_size) # (B, T, 32, 128)
        key = k.view(B, T, n_head, head_size)
        value = v.view(B, T, n_head, head_size)

        query, key = apply_rope(query, key, freqs_cis=freqs_cis)

        # cache token pos (T should be 1 for kv cache aware inference)
        self.cache_k[:B, start_pos:start_pos + T] = key
        self.cache_v[:B, start_pos:start_pos + T] = value

        key = self.cache_k[:B, :start_pos + T]
        value = self.cache_v[:B, :start_pos + T]

        query = query.transpose(1, 2) # (B, 32, T, 128)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        wei = query @ key.transpose(-2, -1) * head_size**-0.5 # (B, 32, T, 128) @ (B, 32, 128, T) = (B, 32, T, start_pos + T)

        if T > 1:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=wei.device))
            wei = wei.masked_fill(~mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ value # (B, 32, T, 128)
        out = out.transpose(1, 2) # (B, T, 32, 128)
        out = out.contiguous().view(B, T, C)
        # mix the heads
        out = self.wo(out)

        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(n_embd * 8 / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        values = self.w3(x)

        return self.w2(gate * values)

# transformer block
class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = Attention()
        self.feed_forward = FeedForward()
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()

    def forward(self, x, freqs_cis, start_pos):
        # residual connections
        x = x + self.attention(self.attention_norm(x), freqs_cis, start_pos)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        freqs_cis = precompute_complex_exponential_freqs(head_size, block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.tok_embeddings = nn.Embedding(tokenizer.vocab_size, n_embd)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(AttentionBlock())

        self.norm = RMSNorm()

        self.output = nn.Linear(n_embd, tokenizer.vocab_size, bias=False)
    
    @torch.inference_mode()
    def forward(self, input, start_pos):
        B, T = input.shape

        tok_emb = self.tok_embeddings(input)
        freqs_cis = self.freqs_cis[start_pos:start_pos + T]

        for layer in self.layers:
            tok_emb = layer(tok_emb, freqs_cis, start_pos)

        tok_emb = self.norm(tok_emb)
        final = self.output(tok_emb)
        return final

        
torch.set_default_dtype(torch.float16)
model = Transformer()
model = model.to(device)

weights = torch.load(
    "./llama-2-7b/consolidated.00.pth",
    weights_only=True,
)

weights.pop("rope.freqs", None)
# load llama2 weights
model.load_state_dict(weights)

input_tokens = tokenizer("hello", return_tensors="pt")["input_ids"].to(device)
max_tokens = 10
for _ in range(max_tokens):
    logits = model(input_tokens)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    input_tokens = torch.cat((input_tokens, idx_next), dim=1)

text = tokenizer.decode(input_tokens[0].cpu())
print(text)

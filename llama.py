import torch
import torch.nn as nn
import torch.nn.functional as F
import json

with open("./llama-2-7b/params.json", "r") as f:
    config = json.load(f)

# hyperparams
block_size = 4096 # i think this is typical for Llama 2 7B
n_embd = config["dim"]
n_head = config["n_heads"]
head_size = n_embd // n_head
n_layers = config["n_layers"]
eps = config["norm_eps"]
multiple_of = config["multiple_of"]

def precompute_complex_exponential_freqs(n_embd, end, theta = 10000.0):
    """
    end = end index
    theta = scaling factor for frequency computation
    split tensor into groups of 2 to perform rotations
    return in complex64 datatype
    a + ib = r(cos(theta) + isin(theta))
    
    """
    # calc rotation freq, make sure n_embd is even
    freqs = 1.0 / (theta ** (torch.arrange(0, n_embd, 2).float() / n_embd)) 
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

def apply_rope(query, key freqs_cis):
    q = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    k = torch.view_as_complex(key.float().reshape(*key.shape[:-1][-1], -1, 2))
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

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape

        q = self.wq(x) # (B, T, 4096)
        k = self.wk(x) 
        v = self.wv(x)

        # split into attention heads
        query = q.view(B, T, n_head, head_size) # (B, T, 32, 128)
        key = k.view(B, T, n_head, head_size)
        value = v.view(B, T, n_head, head_size)

        query = query.transpose(1, 2) # (B, 32, T, 128)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        wei = query @ key.transpose(-2, -1) * head_size**-0.5 # (B, 32, T, 128) @ (B, 32, 128, T) = (B, 32, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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

        self.mah = Attention()
        self.ffwd = FeedForward()
        self.ln1 = RMSNorm()
        self.ln2 = RMSNorm()

    def forward(self, x):
        x = x + self.mah(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

# model = Transformer()

weights = torch.load(
    "./llama-2-7b/consolidated.00.pth",
    weights_only=True,
)



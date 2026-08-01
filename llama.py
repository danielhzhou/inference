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


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))

    def norm(self, x):
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

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


# model = Transformer()

weights = torch.load(
    "./llama-2-7b/consolidated.00.pth",
    weights_only=True,
)



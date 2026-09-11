import torch

def my_layer_norm(x, weight, bias, eps=1e-5):
    m = torch.mean(x, dim=-1, keepdim=True)
    sub = x - m
    variance = torch.mean(torch.pow(sub, 2), dim=-1, keepdim=True)

    return sub / torch.sqrt(variance + eps) * weight + bias


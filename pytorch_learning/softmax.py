import torch

def my_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    sub = x - m
    numerator = torch.exp(sub)

    denominator = torch.sum(numerator, dim=dim, keepdim=True)

    return numerator / denominator

    
    
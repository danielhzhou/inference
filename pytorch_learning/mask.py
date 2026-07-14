import torch

scores = torch.rand(4, 4)
print(scores)
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)  # True above the diagonal = "future"
scores = scores.masked_fill(mask, float('-inf'))

print(scores)
print(mask)

import torch

checkpoint = torch.load(
    "./llama-2-7b/consolidated.00.pth",
    weights_only=True,
)

print(f"num_tensors: {len(checkpoint)}")

for name, tensor in list(checkpoint.items())[:10]:
    print(name, tensor.shape, tensor.dtype)
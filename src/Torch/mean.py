import torch

# Create input tensor (same data as PHP example)
a = torch.tensor([
    [[1, 2], [2, 3]],
    [[1, 4], [1, 1]],
], dtype=torch.float32, requires_grad=True)

print("a data:\n", a)

# Mean over first axis (axis=0)
b = a.mean(dim=-1)

# Backward pass: equivalent to PHP graph runtime forward+backward
b.sum().backward()

print("b data:\n", b)
print("a grad:\n", a.grad)

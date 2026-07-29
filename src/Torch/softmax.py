import torch

a = torch.tensor([[[1, 2], [2, 3]], [[1, 4], [1, 1]]], dtype=torch.float32)
print("a Data:")
print(a)

# enable grads

a = a.clone().detach().requires_grad_(True)

b = torch.softmax(a, dim=-1)

# backward on sum to mimic scalar loss behavior
b.sum().backward()

print("b Data:")
print(b.detach())

print("a Grad:")
print(a.grad)

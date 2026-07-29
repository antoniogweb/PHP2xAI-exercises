import torch

a = torch.tensor([[[1, 2], [2, 3]], [[3, 4], [5, 6]]], dtype=torch.float32)
print("a Data:")
print(a)

b = torch.tensor([1, 2], dtype=torch.float32)
print("b Data:")
print(b)

# enable grads

a = a.clone().detach().requires_grad_(True)
b = b.clone().detach().requires_grad_(True)

c = a + b

# run backward on sum to match scalar loss behavior
c.sum().backward()

print("c Data:")
print(c.detach())

print("b Grad:")
print(b.grad)

print("a Grad:")
print(a.grad)

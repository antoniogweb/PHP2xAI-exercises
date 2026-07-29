import torch

a = torch.tensor(
    [
        [[1.0, 2.0, 3.0], [2.0, 3.0, 3.0]],
        [[3.0, 4.0, 1.0], [5.0, 6.0, 1.0]],
        [[3.0, 4.0, 2.0], [5.0, 6.0, 2.0]],
    ],
    dtype=torch.float32,
    requires_grad=True,
)

b = torch.tensor(
    [
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        [[3.0, 4.0], [5.0, 6.0], [3.0, 4.0]],
        [[3.0, 4.0], [5.0, 6.0], [2.0, 1.0]],
    ],
    dtype=torch.float32,
    requires_grad=True,
)

print("a:")
print(a)
print("b:")
print(b)

c = torch.matmul(a, b)

c.backward(torch.ones_like(c))

print("c:")
print(c)
print("grad b:")
print(b.grad)
print("grad a:")
print(a.grad)

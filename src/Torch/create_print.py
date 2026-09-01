# source ~/venvs/torch/bin/activate
# deactivate

import torch

# Usa float64 per confronti numerici più stabili con la tua libreria
A = torch.tensor([[1.0, 2.0],
                  [2.0, 6.0]], dtype=torch.float64, requires_grad=True)

B = torch.tensor([[1.0, 2.0],
                  [2.0, 3.0]], dtype=torch.float64, requires_grad=True)

C = A @ B  # matmul

print("C.data:\n", C.detach().numpy())

C.backward(torch.ones_like(C))

print("B.grad:\n", B.grad.detach().numpy())
print("A.grad:\n", A.grad.detach().numpy())
import torch
import torch.nn.functional as functional


def run_gelu(name: str, values: list) -> None:
    input_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    output = functional.gelu(input_tensor, approximate="tanh")
    output.backward(torch.ones_like(output))

    print(f"\n=== {name} ===")
    print("output:")
    print(output)
    print("input grad:")
    print(input_tensor.grad)


# approximate="tanh" matches PHP2xAI's GELU implementation.
run_gelu(
    "Matrix: negative, zero and positive values",
    [[-3.0, -1.0, 0.0], [1.0, 2.0, 3.0]],
)
run_gelu(
    "Token embeddings",
    [[[-2.0, -0.5], [0.5, 2.0]], [[-1.5, 0.0], [1.5, 3.0]]],
)

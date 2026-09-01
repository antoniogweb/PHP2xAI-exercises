import torch


def run_scale(name: str, values: list, scale: float) -> None:
    input_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    output = input_tensor * scale
    output.backward(torch.ones_like(output))

    print(f"\n=== {name} ===")
    print("scale:", scale)
    print("output:")
    print(output)
    print("input grad:")
    print(input_tensor.grad)


run_scale("Positive scalar on a matrix", [[1.0, -2.0], [3.0, -4.0]], 2.5)
run_scale(
    "Negative scalar on token embeddings",
    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    -0.5,
)
run_scale("Zero scalar", [1.0, -2.0, 3.0], 0.0)

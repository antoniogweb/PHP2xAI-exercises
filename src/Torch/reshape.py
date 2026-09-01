import torch


def run_reshape(name: str, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
    input_tensor = torch.arange(
        1, torch.tensor(input_shape).prod().item() + 1, dtype=torch.float32
    ).reshape(input_shape).requires_grad_()
    output = input_tensor.reshape(output_shape)
    output.backward(torch.ones_like(output))

    print(f"\n=== {name} ===")
    print("input shape: ", tuple(input_tensor.shape))
    print("output shape:", tuple(output.shape))
    print("output:")
    print(output)
    print("input grad:")
    print(input_tensor.grad)


# Equivalent to PHP2xAI [2, 3] -> [3, 2].
run_reshape("2D reshape", (2, 3), (3, 2))

# PyTorch uses -1 with the same inference semantics as PHP2xAI.
run_reshape("Inferred dimension -1", (2, 2, 3), (2, -1))

# Typical multi-head attention merge: [B, L, H, dk] -> [B, L, D].
run_reshape("Merge attention heads", (1, 3, 2, 2), (1, 3, -1))

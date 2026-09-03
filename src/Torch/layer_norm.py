import torch
import torch.nn.functional as functional


def run_layer_norm_last_axis(name: str, values: list, gamma: list, beta: list) -> None:
    input_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
    beta_tensor = torch.tensor(beta, dtype=torch.float32, requires_grad=True)

    output = functional.layer_norm(
        input_tensor,
        normalized_shape=(input_tensor.shape[-1],),
        weight=gamma_tensor,
        bias=beta_tensor,
        eps=1.0e-5,
    )
    output.backward(torch.ones_like(output))

    print(f"\n=== {name} ===")
    print("output:")
    print(output)
    print("input grad:")
    print(input_tensor.grad)
    print("gamma grad:")
    print(gamma_tensor.grad)
    print("beta grad:")
    print(beta_tensor.grad)


def run_layer_norm_axis(name: str, values: list, gamma: list, beta: list, axis: int) -> None:
    input_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
    beta_tensor = torch.tensor(beta, dtype=torch.float32, requires_grad=True)

    # PyTorch LayerNorm acts on trailing axes. Move the requested axis to the
    # end, apply LayerNorm, then restore the original layout.
    moved = input_tensor.movedim(axis, -1)
    normalized = functional.layer_norm(
        moved,
        normalized_shape=(input_tensor.shape[axis],),
        weight=gamma_tensor,
        bias=beta_tensor,
        eps=1.0e-5,
    )
    output = normalized.movedim(-1, axis)
    output.backward(torch.ones_like(output))

    print(f"\n=== {name} ===")
    print("output:")
    print(output)
    print("input grad:")
    print(input_tensor.grad)
    print("gamma grad:")
    print(gamma_tensor.grad)
    print("beta grad:")
    print(beta_tensor.grad)


run_layer_norm_last_axis(
    "Transformer tokens [B, L, D], last axis",
    [[[1.0, -2.0, 3.5], [4.0, 0.5, -1.0]], [[2.0, 8.0, -3.0], [1.5, 2.5, 6.0]]],
    [1.2, -0.7, 0.4],
    [0.1, 0.2, -0.3],
)

run_layer_norm_axis(
    "Generic axis 1 on [B, H, L, D]",
    [
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[2.0, -1.0], [0.5, 3.0]],
        ],
        [
            [[-2.0, 1.0], [4.0, -3.0]],
            [[0.0, 2.0], [1.0, 5.0]],
            [[3.0, 4.0], [-1.0, 2.0]],
        ],
    ],
    [1.0, 0.5, -0.25],
    [0.0, 0.1, -0.2],
    axis=1,
)

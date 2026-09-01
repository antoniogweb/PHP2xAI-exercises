import torch


def run_transpose(name: str, shape: tuple[int, ...], axes: tuple[int, int] = (-2, -1)) -> None:
    input_tensor = torch.arange(
        1, torch.tensor(shape).prod().item() + 1, dtype=torch.float32
    ).reshape(shape).requires_grad_()
    output = input_tensor.transpose(*axes)
    output.backward(torch.ones_like(output))

    print(f"\n=== {name} ===")
    print("input shape: ", tuple(input_tensor.shape))
    print("output shape:", tuple(output.shape))
    print("output:")
    print(output)
    print("input grad:")
    print(input_tensor.grad)


# Equivalent to PHP2xAI TRANSPOSE_2D.
run_transpose("2D, default axes (-2, -1)", (2, 3))

# Equivalent to PHP2xAI TRANSPOSE_3D_LAST_TWO.
run_transpose("3D, default axes (-2, -1)", (2, 2, 3))

# Multi-head attention layout:
# [B, L, H, dk] -> [B, H, L, dk] -> [B, H, dk, L].
attention_input = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 2, 2).requires_grad_()
by_head = attention_input.transpose(1, 2)
for_scores = by_head.transpose(-2, -1)
for_scores.backward(torch.ones_like(for_scores))

print("\n=== Multi-head attention transpose ===")
print("[B, L, H, dk]:", tuple(attention_input.shape))
print("[B, H, L, dk]:", tuple(by_head.shape))
print("[B, H, dk, L]:", tuple(for_scores.shape))
print(for_scores)
print("input grad:")
print(attention_input.grad)

# Equivalent to PHP2xAI TRANSPOSE_GENERIC.
run_transpose("Generic axes (0, 2)", (2, 2, 3, 2), (0, 2))

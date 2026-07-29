import torch
import torch.nn.functional as F


def print_case(title, logits_data, target_data, axis):
    logits = torch.tensor(logits_data, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(target_data, dtype=torch.long)

    logits_for_ce = logits if axis == 1 else logits.movedim(axis, 1)
    loss = F.cross_entropy(logits_for_ce, target, reduction="none")

    loss.sum().backward()

    print(f"\n=== {title} ===")
    print("Logits:")
    print(logits.detach())
    print("Target label int:")
    print(target)
    print("Loss:")
    print(loss.detach())
    print("Logits grad:")
    print(logits.grad)


print_case(
    "3D last axis",
    [
        [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 1.0], [3.0, 1.0, -2.0]],
    ],
    [
        [2, 1],
        [0, 0],
    ],
    axis=-1,
)

print_case(
    "3D generic axis 1",
    [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
    ],
    [
        [2, 0],
        [1, 2],
    ],
    axis=1,
)

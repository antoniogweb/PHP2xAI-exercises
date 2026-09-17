import torch


def assert_values(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{label}: mismatch")


lq = 3
lkv = 5
scores = torch.arange(1, 31, dtype=torch.float32).reshape(1, 2, lq, lkv)
scores.requires_grad_()

offset = lkv - lq
future_mask = torch.triu(
    torch.ones((lq, lkv), dtype=torch.bool), diagonal=offset + 1
)
output = scores.masked_fill(future_mask.view(1, 1, lq, lkv), float("-inf"))
output.backward(torch.ones_like(output))

expected_output = scores.detach().clone()
expected_output.masked_fill_(future_mask.view(1, 1, lq, lkv), float("-inf"))
expected_grad = (~future_mask).to(torch.float32).view(1, 1, lq, lkv).expand_as(scores)

assert_values("forward", output.detach(), expected_output)
assert_values("backward", scores.grad, expected_grad)

print("Torch causal-mask forward/backward: OK")

"""Compare PHP2xAI RoPE (PHP and C++ runtimes) with PyTorch autograd.

Run from this directory or any other directory:
    python3 src/Torch/rope.py
"""

import json
import subprocess
import sys
from pathlib import Path

import torch


TOLERANCE = 1e-5


def rope(x, position_axis, rotation_axis, offset, base, pairing):
    """RoPE reference using Torch operators, preserving the original layout."""
    rank = x.ndim
    position_axis %= rank
    rotation_axis %= rank
    outer_axes = [axis for axis in range(rank) if axis not in (position_axis, rotation_axis)]
    permutation = outer_axes + [position_axis, rotation_axis]
    inverse_permutation = [permutation.index(axis) for axis in range(rank)]

    work = x.permute(permutation)
    length, dim = work.shape[-2:]
    half_dim = dim // 2
    positions = torch.arange(offset, offset + length, dtype=x.dtype, device=x.device)
    frequencies = base ** (-2.0 * torch.arange(half_dim, dtype=x.dtype, device=x.device) / dim)
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = angles.cos(), angles.sin()
    output = torch.empty_like(work)

    if pairing == "INTERLEAVED":
        first, second = work[..., 0::2], work[..., 1::2]
        output[..., 0::2] = first * cosine - second * sine
        output[..., 1::2] = first * sine + second * cosine
    elif pairing == "ROTATE_HALF":
        first, second = work[..., :half_dim], work[..., half_dim:]
        output[..., :half_dim] = first * cosine - second * sine
        output[..., half_dim:] = first * sine + second * cosine
    else:
        raise ValueError(f"Unsupported pairing: {pairing}")

    return output.permute(inverse_permutation)


def max_difference(actual, expected):
    return (actual - expected).abs().max().item()


def main():
    php_script = Path(__file__).resolve().parents[1] / "PHP" / "rope.php"
    completed = subprocess.run(["php", str(php_script), "--json"], check=True, capture_output=True, text=True)
    cases = json.loads(completed.stdout)

    for case in cases:
        source = torch.tensor(case["input"], dtype=torch.float32).reshape(case["shape"]).requires_grad_(True)
        output = rope(source, case["position_axis"], case["rotation_axis"], case["offset"], case["base"], case["pairing"])
        output.sum().backward()

        torch_output = output.detach().flatten()
        torch_grad = source.grad.flatten()
        php_output = torch.tensor(case["php_output"], dtype=torch.float32)
        cpp_output = torch.tensor(case["cpp_output"], dtype=torch.float32)
        php_grad = torch.tensor(case["php_grad"], dtype=torch.float32)
        cpp_grad = torch.tensor(case["cpp_grad"], dtype=torch.float32)

        differences = {
            "PHP output": max_difference(php_output, torch_output),
            "C++ output": max_difference(cpp_output, torch_output),
            "PHP gradient": max_difference(php_grad, torch_grad),
            "C++ gradient": max_difference(cpp_grad, torch_grad),
        }
        if any(value > TOLERANCE for value in differences.values()):
            raise AssertionError(f"{case['name']}: {differences}")

        print(f"{case['name']}: {differences}")

    print("PHP, C++ and PyTorch RoPE forward/backward: OK")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            sys.exit("PyTorch is required: install it with `python3 -m pip install torch`.")
        raise

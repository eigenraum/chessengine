"""Shared device selection (DESIGN-GPU.md section 3): auto picks cuda, then
Apple Silicon (mps), then cpu. Used by TorchEvaluator and every entry point
that constructs one (train, self-play, arena, the CLI/web UIs) so the
`--device auto|cpu|cuda|mps` flag means the same thing everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEVICE_CHOICES = ["auto", "cpu", "cuda", "mps"]


def select_device(requested: str = "auto") -> "torch.device":
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: "torch.device") -> str:
    import torch

    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"

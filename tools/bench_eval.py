"""Micro-benchmark for TorchEvaluator across devices and batch sizes
(DESIGN-GPU.md section 4.4). Not a test — a number producer for deciding how
urgent G2 (cross-engine batching) is.

    uv run python tools/bench_eval.py
    uv run python tools/bench_eval.py --devices cpu cuda --batch-sizes 1 64 256
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from chessengine import _mcts
from chessengine.eval.device import DEVICE_CHOICES, describe_device, select_device
from chessengine.eval.torch_eval import TorchEvaluator

_DEFAULT_BATCH_SIZES = [1, 8, 64, 256]
_WARMUP = 5
_ITERS = 20


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def bench_one(evaluator: TorchEvaluator, batch_size: int) -> float:
    """Returns mean wall-clock seconds per __call__ over _ITERS calls."""
    rng = np.random.default_rng(0)
    planes = rng.standard_normal((batch_size, _mcts.PLANES, 8, 8)).astype(np.float32)

    for _ in range(_WARMUP):
        evaluator(planes)
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(_ITERS):
        evaluator(planes)
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / _ITERS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=DEVICE_CHOICES, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=_DEFAULT_BATCH_SIZES)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--filters", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    devices = args.devices or available_devices()

    print(f"{'device':<10} {'batch':>6} {'ms/call':>10} {'positions/s':>12}")
    for device_name in devices:
        device = select_device(device_name)
        evaluator = TorchEvaluator(blocks=args.blocks, filters=args.filters, device=device_name)
        for batch_size in args.batch_sizes:
            seconds = bench_one(evaluator, batch_size)
            print(
                f"{describe_device(device):<10} {batch_size:>6} {seconds * 1000:>10.3f} "
                f"{batch_size / seconds:>12.0f}"
            )


if __name__ == "__main__":
    main()

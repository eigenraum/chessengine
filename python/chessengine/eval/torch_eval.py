"""AlphaZero-style policy/value net and the evaluator callback wrapping it
(DESIGN-M6.md section 6). Optional dependency: torch is imported only here,
never at `chessengine` package import time — the material evaluator must
keep working in installs without the `train` dependency group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from chessengine import _mcts
from chessengine.eval.device import select_device

FILTERS_DEFAULT = 64


class _ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class PolicyValueNet(nn.Module):
    """Small AlphaZero-style ResNet: conv trunk, policy head, value head.

    forward(planes: float32 [N, 19, 8, 8]) -> (values [N] in (0, 1),
    policy_logits [N, 4672] — raw, unmasked; the search applies legal-move
    masking and softmax, not the net (DESIGN-M6.md section 4.1/6)).
    """

    def __init__(self, blocks: int = 4, filters: int = FILTERS_DEFAULT) -> None:
        super().__init__()
        # Stored for checkpointing: the file alone must reconstruct the
        # architecture (see TorchEvaluator).
        self.blocks = blocks
        self.filters = filters

        self.stem_conv = nn.Conv2d(_mcts.PLANES, filters, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(filters)
        self.trunk = nn.ModuleList(_ResidualBlock(filters) for _ in range(blocks))

        self.policy_conv = nn.Conv2d(filters, 73, 1)

        self.value_conv = nn.Conv2d(filters, 8, 1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.stem_bn(self.stem_conv(planes)))
        for block in self.trunk:
            x = block(x)

        # Flattening [N, 73, 8, 8] yields index plane*64 + square, exactly
        # eval::move_index's convention (DESIGN-M6.md section 3.3).
        policy_logits = self.policy_conv(x).flatten(1)

        v = torch.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.sigmoid(self.value_fc2(v)).squeeze(-1)

        return value, policy_logits


class TorchEvaluator:
    """Engine evaluator callback backed by PolicyValueNet.

    __call__(planes: np.ndarray [N,19,8,8]) -> (values [N], logits [N,4672]),
    both float32 numpy. Runs on the C++ evaluator thread under the GIL
    (PyEvaluator in bindings.cpp) — one deliberate Python round-trip per
    batch (DESIGN-GPU.md section 4); `device` picks where the forward pass
    itself runs, `cpu` by default so library use and tests stay on the
    deterministic reference path unless a caller opts in.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        blocks: int = 4,
        filters: int = FILTERS_DEFAULT,
        device: str = "cpu",
    ) -> None:
        self.device = select_device(device)
        if self.device.type == "cpu":
            # Self-play/arena get their parallelism from OS processes
            # (--jobs), not from torch threads within one; on cuda/mps this
            # doesn't apply (the CPU here only encodes/copies).
            torch.set_num_threads(1)
        if checkpoint is not None:
            data = torch.load(checkpoint, map_location="cpu")
            self.model = PolicyValueNet(blocks=data["blocks"], filters=data["filters"])
            self.model.load_state_dict(data["state_dict"])
        else:
            self.model = PolicyValueNet(blocks=blocks, filters=filters)
        self.model.eval().to(self.device)

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            x = torch.from_numpy(planes).to(self.device, non_blocking=True)
            values, logits = self.model(x)
        return (
            values.cpu().numpy().astype(np.float32),
            logits.cpu().numpy().astype(np.float32),
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "blocks": self.model.blocks,
                "filters": self.model.filters,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

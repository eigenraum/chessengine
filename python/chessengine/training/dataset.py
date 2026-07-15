"""Self-play data: one `.npz` shard per game, plus the training-side window
and batch sampling (DESIGN-M6.md section 7.2/7.3). Shard I/O and row
filtering are plain numpy — no torch import here at module level, so
selfplay.py's writer and pytest can both use them without the `train`
dependency group. Batch/sample_batch import torch lazily, only when called.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from chessengine import _mcts


@dataclass
class Row:
    """One exported search-tree node, ready to write to a shard.

    `outcome` is filled in after the game ends (DESIGN-M6.md section 7.2):
    the final result from this row's side-to-move perspective.
    """

    fen: str
    policy_index: list[int]
    policy_prob: list[float]
    search_value: float
    visit_count: int
    is_root: bool
    outcome: float = 0.0


@dataclass
class Shard:
    """One game's rows as flat parallel arrays (the on-disk shard layout).

    Row i's sparse policy target is `policy_index[row_offsets[i]:row_offsets[i+1]]`
    / the matching slice of `policy_prob`.
    """

    fens: np.ndarray  # object dtype, str
    policy_index: np.ndarray  # int32
    policy_prob: np.ndarray  # float32
    row_offsets: np.ndarray  # int64, len(fens) + 1
    search_value: np.ndarray  # float32
    visit_count: np.ndarray  # uint32
    is_root: np.ndarray  # bool
    outcome: np.ndarray  # float32
    meta: dict

    def __len__(self) -> int:
        return len(self.fens)


def save_game_shard(path: str | Path, rows: list[Row], meta: dict) -> None:
    row_offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, row in enumerate(rows):
        row_offsets[i + 1] = row_offsets[i] + len(row.policy_index)

    if rows:
        policy_index = np.concatenate(
            [np.asarray(r.policy_index, dtype=np.int32) for r in rows]
        )
        policy_prob = np.concatenate(
            [np.asarray(r.policy_prob, dtype=np.float32) for r in rows]
        )
    else:
        policy_index = np.zeros(0, dtype=np.int32)
        policy_prob = np.zeros(0, dtype=np.float32)

    np.savez(
        path,
        fens=np.array([r.fen for r in rows], dtype=object),
        policy_index=policy_index,
        policy_prob=policy_prob,
        row_offsets=row_offsets,
        search_value=np.asarray([r.search_value for r in rows], dtype=np.float32),
        visit_count=np.asarray([r.visit_count for r in rows], dtype=np.uint32),
        is_root=np.asarray([r.is_root for r in rows], dtype=bool),
        outcome=np.asarray([r.outcome for r in rows], dtype=np.float32),
        meta=json.dumps(meta),
    )


def load_shard(path: str | Path) -> Shard:
    with np.load(path, allow_pickle=True) as data:
        return Shard(
            fens=data["fens"],
            policy_index=data["policy_index"],
            policy_prob=data["policy_prob"],
            row_offsets=data["row_offsets"],
            search_value=data["search_value"],
            visit_count=data["visit_count"],
            is_root=data["is_root"],
            outcome=data["outcome"],
            meta=json.loads(str(data["meta"])),
        )


def load_window(data_dir: str | Path, window: int) -> list[Shard]:
    """The newest `window` game shards in `data_dir`, oldest first."""
    paths = sorted(Path(data_dir).glob("game-*.npz"), key=lambda p: p.stat().st_mtime)
    return [load_shard(p) for p in paths[-window:]]


class FilteredRow(NamedTuple):
    shard_index: int
    row_index: int
    value_target: float


def filter_rows(
    shards: list[Shard],
    lambda_root: float = 1.0,
    lambda_interior: float = 0.0,
    min_visits_interior: int = 32,
) -> list[FilteredRow]:
    """Rows kept for training, with their blended value target precomputed
    (DESIGN-M6.md section 7.3): roots always kept (lambda_root blend);
    interior rows kept only with visit_count >= min_visits_interior
    (lambda_interior blend). Call once per window — sample_batch reuses the
    result across many steps instead of re-filtering every call.
    """
    rows: list[FilteredRow] = []
    for si, shard in enumerate(shards):
        for ri in range(len(shard)):
            is_root = bool(shard.is_root[ri])
            if not is_root and shard.visit_count[ri] < min_visits_interior:
                continue
            lam = lambda_root if is_root else lambda_interior
            target = lam * shard.outcome[ri] + (1.0 - lam) * shard.search_value[ri]
            rows.append(FilteredRow(si, ri, float(target)))
    return rows


@dataclass
class Batch:
    # Types are torch.Tensor; left unannotated-at-runtime (deferred by
    # `from __future__ import annotations`) so importing this module never
    # requires torch — only sample_batch(), which lazy-imports it, does.
    planes: object  # float32 [B, 19, 8, 8]
    policy_index: object  # int64 [B, Kmax], 0-padded
    policy_prob: object  # float32 [B, Kmax], 0-padded
    value_target: object  # float32 [B]


def _batch_from_indices(
    shards: list[Shard], rows: list[FilteredRow], indices: np.ndarray,
):
    """Shared by sample_batch (random, with replacement) and
    iterate_batches (deterministic full pass, for validation)."""
    import torch

    n = len(indices)
    planes = np.empty((n, _mcts.PLANES, 8, 8), dtype=np.float32)
    value_target = np.empty(n, dtype=np.float32)
    idx_slices: list[np.ndarray] = []
    prob_slices: list[np.ndarray] = []

    for b, r in enumerate(indices):
        shard_index, row_index, target = rows[r]
        shard = shards[shard_index]
        planes[b] = _mcts.encode_planes(str(shard.fens[row_index]))
        value_target[b] = target
        start, end = shard.row_offsets[row_index], shard.row_offsets[row_index + 1]
        idx_slices.append(shard.policy_index[start:end])
        prob_slices.append(shard.policy_prob[start:end])

    kmax = max(len(idx) for idx in idx_slices)
    # Padding index 0 (never -1): gather() would crash on an out-of-range
    # index, and padding prob 0 makes padded entries contribute nothing to
    # the policy loss regardless of which valid index they point at.
    policy_index = np.zeros((n, kmax), dtype=np.int64)
    policy_prob = np.zeros((n, kmax), dtype=np.float32)
    for b, (idx, prob) in enumerate(zip(idx_slices, prob_slices)):
        policy_index[b, : len(idx)] = idx
        policy_prob[b, : len(prob)] = prob

    return Batch(
        planes=torch.from_numpy(planes),
        policy_index=torch.from_numpy(policy_index),
        policy_prob=torch.from_numpy(policy_prob),
        value_target=torch.from_numpy(value_target),
    )


def sample_batch(
    shards: list[Shard],
    batch_size: int,
    rng: np.random.Generator,
    lambda_root: float = 1.0,
    lambda_interior: float = 0.0,
    min_visits_interior: int = 32,
    rows: list[FilteredRow] | None = None,
) -> Batch:
    """Uniformly-sampled minibatch (with replacement), planes recomputed
    through `_mcts.encode_planes` (never a duplicated Python encoder — the
    train/search encoding-identity guarantee, DESIGN-M6.md section 3.4).

    `rows`: pass the result of `filter_rows(shards, ...)` when calling this
    in a loop (a training step) to avoid re-filtering the whole window every
    time; computed on demand otherwise.
    """
    if rows is None:
        rows = filter_rows(shards, lambda_root, lambda_interior, min_visits_interior)
    if not rows:
        raise ValueError("no rows survive the filter (empty window or min_visits_interior too strict)")

    chosen = rng.integers(0, len(rows), size=batch_size)
    return _batch_from_indices(shards, rows, chosen)


def iterate_batches(shards: list[Shard], rows: list[FilteredRow], batch_size: int):
    """Deterministic, non-overlapping full pass over `rows` (last batch may
    be smaller). For validation-loss averaging, where sample_batch's
    with-replacement sampling would double-count rows and skip others.
    """
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        yield _batch_from_indices(shards, rows, np.arange(start, end))


def stream_shard_batches(
    shard_paths: Sequence[str | Path],
    batch_size: int,
    rng: np.random.Generator,
    *,
    buffer_shards: int = 8,
    shuffle: bool = True,
    lambda_root: float = 1.0,
    lambda_interior: float = 0.0,
    min_visits_interior: int = 32,
):
    """One epoch of batches over `shard_paths`, loading only `buffer_shards`
    shards into memory at a time — for datasets far larger than RAM (the
    supervised PGN corpus, pretrain.py) where load_window's "all shards
    resident" model does not fit.

    Batches never span buffers, so `buffer_shards` is the shuffling grain: a
    small buffer streams with little memory but mixes rows only within a few
    shards; a large one shuffles more thoroughly at higher memory cost. With
    `shuffle=True` both the buffer order and the rows within each buffer are
    permuted (deterministically, from `rng`); `shuffle=False` is a stable
    full pass, for validation. Encoding goes through the same
    `_batch_from_indices` (hence `_mcts.encode_planes`) as training and
    self-play — one encoder everywhere (DESIGN-M6.md section 3.4).
    """
    order = np.arange(len(shard_paths))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), buffer_shards):
        shards = [load_shard(shard_paths[i]) for i in order[start : start + buffer_shards]]
        rows = filter_rows(shards, lambda_root, lambda_interior, min_visits_interior)
        if not rows:
            continue
        indices = np.arange(len(rows))
        if shuffle:
            rng.shuffle(indices)
        for b in range(0, len(indices), batch_size):
            yield _batch_from_indices(shards, rows, indices[b : b + batch_size])

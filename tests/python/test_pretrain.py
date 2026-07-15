"""Supervised PGN import + pretraining tests (data_pretrain/README.md).

Tests 1-2 exercise pgn_import.py's parsing/filtering and the shard it writes;
they use numpy + the C++ encoders only, no torch, so they run in the default
install. Test 3 needs torch (importorskip inside the test) and uses a tiny
net/PGN — a plumbing smoke gate, matching test_training.py's style.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chessengine.training.dataset import load_shard
from chessengine.training.pgn_import import run as import_run

# Game A: standard, both rated >= 1500, decisive, 10 plies -> kept by defaults.
# Game B: standard, low-rated, drawn, 6 plies -> dropped by --min-elo 1500.
# Game C: Chess960 (Variant + FEN header) -> dropped as non-standard.
# Game D: unfinished ("*" result) -> dropped (no usable outcome).
SAMPLE_PGN = """[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "Rated Blitz game"]
[Result "1/2-1/2"]
[WhiteElo "1200"]
[BlackElo "1250"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 1/2-1/2

[Event "Rated Blitz game"]
[Variant "Chess960"]
[FEN "nrbbqnkr/pppppppp/8/8/8/8/PPPPPPPP/NRBBQNKR w KQkq - 0 1"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]

1. Nb3 Nb6 2. c4 c5 1-0

[Event "Rated Blitz game"]
[Result "*"]
[WhiteElo "1900"]
[BlackElo "1900"]

1. e4 e5 *
"""


def _write_pgn(tmp_path):
    pgn = tmp_path / "games.pgn"
    pgn.write_text(SAMPLE_PGN)
    return pgn


# --- 1. Import with default filters ----------------------------------------


def test_pgn_import_default_filters_keep_only_game_a(tmp_path):
    out = tmp_path / "shards"
    paths = import_run(["--pgn", str(_write_pgn(tmp_path)), "--out", str(out)])

    # Only game A survives min-elo 1500 + standard-only + finished-result.
    assert len(paths) == 1
    shard = load_shard(paths[0])
    assert len(shard) == 10  # one row per ply of game A

    # Every row is a one-hot played move recorded as a search root.
    assert shard.is_root.all()
    np.testing.assert_array_equal(shard.row_offsets, np.arange(11))  # one policy entry per row
    np.testing.assert_allclose(shard.policy_prob, np.ones(10), atol=1e-6)
    assert ((shard.policy_index >= 0) & (shard.policy_index < 4672)).all()

    # White won: white-to-move rows target 1.0, black-to-move rows 0.0, and
    # outcome == search_value (there is no search in supervised data).
    sides = np.array([fen.split()[1] for fen in shard.fens])
    np.testing.assert_allclose(shard.outcome[sides == "w"], 1.0)
    np.testing.assert_allclose(shard.outcome[sides == "b"], 0.0)
    np.testing.assert_allclose(shard.outcome, shard.search_value)
    assert shard.meta["supervised"] is True


# --- 2. Filters relaxed -----------------------------------------------------


def test_pgn_import_relaxed_filters_keep_two_games(tmp_path):
    out = tmp_path / "shards"
    paths = import_run(
        [
            "--pgn", str(_write_pgn(tmp_path)), "--out", str(out),
            "--min-elo", "0", "--min-plies", "1", "--games-per-shard", "1",
        ]
    )
    # A and B kept (standard, finished); C (variant) and D ("*") still dropped.
    # games-per-shard 1 -> one shard each.
    assert len(paths) == 2
    total_rows = sum(len(load_shard(p)) for p in paths)
    assert total_rows == 10 + 6

    # Game B was a draw: both colours target 0.5.
    draw_shard = next(load_shard(p) for p in paths if len(load_shard(p)) == 6)
    np.testing.assert_allclose(draw_shard.outcome, 0.5)


# --- 3. Pretraining smoke ---------------------------------------------------


def test_pretrain_smoke(tmp_path):
    pytest.importorskip("torch")
    from chessengine.eval.torch_eval import TorchEvaluator
    from chessengine.training.pretrain import run as pretrain_run

    shards = tmp_path / "shards"
    import_run(
        [
            "--pgn", str(_write_pgn(tmp_path)), "--out", str(shards),
            "--min-elo", "0", "--min-plies", "1", "--games-per-shard", "1",
        ]
    )

    out_path = tmp_path / "pretrained.pt"
    losses = pretrain_run(
        [
            "--data", str(shards), "--out", str(out_path),
            "--blocks", "1", "--filters", "8", "--epochs", "1", "--batch", "8",
            "--val-frac", "0.5", "--buffer-shards", "2", "--device", "cpu",
        ]
    )
    assert out_path.exists()
    assert math.isfinite(losses["value_loss"])
    assert math.isfinite(losses["policy_loss"])
    # val-frac 0.5 over two shards holds one out -> validation ran.
    assert math.isfinite(losses["val_value_loss"])

    reloaded = TorchEvaluator(checkpoint=out_path)
    assert reloaded.model.blocks == 1
    assert reloaded.model.filters == 8


def test_pretrain_no_shards_errors(tmp_path):
    from chessengine.training.pretrain import run as pretrain_run

    with pytest.raises(SystemExit):
        pretrain_run(["--data", str(tmp_path), "--out", str(tmp_path / "x.pt")])

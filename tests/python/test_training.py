"""Self-play, training, and arena tests (DESIGN-M6.md section 7).

Tests 1-3 exercise dataset.py's shard I/O and row-filtering logic directly
and need no torch — they must keep passing in the default (torch-less)
install. Tests 4+ need torch (`pytest.importorskip` inside each test, not
at module level, so it doesn't also skip tests 1-3) and use tiny
nets/games/steps: a plumbing smoke gate, not a claim about strength — see
chessengine/training/__init__.py's docstring for the real acceptance run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chessengine.training.arena import a_plays_white, score_for_a
from chessengine.training.dataset import Row, Shard, filter_rows, load_shard, save_game_shard
from chessengine.training.selfplay import row_outcome, white_result_from_outcome

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"


# --- 1. Shard round-trip (no torch) ----------------------------------------


def test_shard_round_trip(tmp_path):
    rows = [
        Row(
            fen=STARTPOS, policy_index=[76, 3718], policy_prob=[0.6, 0.4],
            search_value=0.55, visit_count=100, is_root=True, outcome=1.0,
        ),
        Row(
            fen=BLACK_TO_MOVE, policy_index=[5], policy_prob=[1.0],
            search_value=0.3, visit_count=10, is_root=False, outcome=0.0,
        ),
    ]
    path = tmp_path / "game-1-1-0.npz"
    save_game_shard(path, rows, meta={"net": "x.pt", "sims": 32})

    shard = load_shard(path)
    assert list(shard.fens) == [STARTPOS, BLACK_TO_MOVE]
    assert len(shard) == 2
    assert list(shard.row_offsets) == [0, 2, 3]
    np.testing.assert_array_equal(shard.policy_index, [76, 3718, 5])
    np.testing.assert_allclose(shard.policy_prob, [0.6, 0.4, 1.0], atol=1e-6)
    np.testing.assert_allclose(shard.search_value, [0.55, 0.3], atol=1e-6)
    np.testing.assert_array_equal(shard.visit_count, [100, 10])
    np.testing.assert_array_equal(shard.is_root, [True, False])
    np.testing.assert_array_equal(shard.outcome, [1.0, 0.0])
    assert shard.meta == {"net": "x.pt", "sims": 32}

    # Ragged alignment: row 1's slice is exactly its own single-entry policy.
    start, end = shard.row_offsets[1], shard.row_offsets[2]
    assert list(shard.policy_index[start:end]) == [5]
    assert list(shard.policy_prob[start:end]) == [1.0]


# --- 2. Outcome perspective (no torch) -------------------------------------


class _FakeOutcome:
    def __init__(self, winner):
        self.winner = winner


def test_outcome_perspective_white_won():
    white_result = white_result_from_outcome(_FakeOutcome(winner=True))
    assert white_result == 1.0
    assert row_outcome(STARTPOS, white_result) == 1.0  # "w" row
    assert row_outcome(BLACK_TO_MOVE, white_result) == 0.0  # "b" row


def test_outcome_perspective_black_won():
    white_result = white_result_from_outcome(_FakeOutcome(winner=False))
    assert white_result == 0.0
    assert row_outcome(STARTPOS, white_result) == 0.0
    assert row_outcome(BLACK_TO_MOVE, white_result) == 1.0


def test_outcome_perspective_draw_and_ply_cap():
    assert white_result_from_outcome(_FakeOutcome(winner=None)) == 0.5
    assert white_result_from_outcome(None) == 0.5  # ply-cap adjudication


# --- 3. Value-target blend (no torch) --------------------------------------


def _make_shard(is_root, visit_count, search_value, outcome) -> Shard:
    n = len(is_root)
    return Shard(
        fens=np.array([STARTPOS] * n, dtype=object),
        policy_index=np.zeros(n, dtype=np.int32),
        policy_prob=np.ones(n, dtype=np.float32),
        row_offsets=np.arange(n + 1, dtype=np.int64),
        search_value=np.asarray(search_value, dtype=np.float32),
        visit_count=np.asarray(visit_count, dtype=np.uint32),
        is_root=np.asarray(is_root, dtype=bool),
        outcome=np.asarray(outcome, dtype=np.float32),
        meta={},
    )


def test_value_target_root_gets_outcome_interior_gets_search_value():
    shard = _make_shard(
        is_root=[True, False], visit_count=[999, 50], search_value=[0.4, 0.7], outcome=[1.0, 1.0]
    )
    targets = {ri: t for _, ri, t in filter_rows([shard], 1.0, 0.0, min_visits_interior=32)}
    assert targets[0] == pytest.approx(1.0)  # lambda_root=1: root row target is outcome
    assert targets[1] == pytest.approx(0.7)  # lambda_interior=0: interior target is search_value


def test_value_target_blend_interpolates():
    shard = _make_shard(
        is_root=[True, False], visit_count=[999, 50], search_value=[0.4, 0.7], outcome=[1.0, 1.0]
    )
    targets = {ri: t for _, ri, t in filter_rows([shard], 0.5, 0.5, min_visits_interior=32)}
    assert targets[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.4)
    assert targets[1] == pytest.approx(0.5 * 1.0 + 0.5 * 0.7)


def test_value_target_min_visits_filter_drops_interior_row():
    shard = _make_shard(
        is_root=[True, False], visit_count=[999, 10], search_value=[0.4, 0.7], outcome=[1.0, 1.0]
    )
    kept = [ri for _, ri, _ in filter_rows([shard], 1.0, 0.0, min_visits_interior=32)]
    assert kept == [0]  # interior row's visit_count (10) < min_visits_interior (32); root always kept


# --- 4. Self-play smoke -----------------------------------------------------


def test_selfplay_smoke(tmp_path):
    pytest.importorskip("torch")
    from chessengine.eval.torch_eval import TorchEvaluator
    from chessengine.training.selfplay import run as selfplay_run

    net_path = tmp_path / "net.pt"
    TorchEvaluator(blocks=1, filters=8).save(net_path)
    out_dir = tmp_path / "data"

    paths = selfplay_run(
        [
            "--net", str(net_path), "--out", str(out_dir),
            "--games", "1", "--sims", "32", "--workers", "1", "--batch-size", "8",
            "--max-plies", "20", "--snapshot-min-visits", "1",
        ]
    )
    assert len(paths) == 1
    assert paths[0].exists()

    shard = load_shard(paths[0])
    assert len(shard) > 0
    # Exactly one root row is recorded per move played (row 0 of every
    # per-move snapshot); the net's random initialization isn't seeded, so
    # the exact ply count isn't pinned here, only its plausible range.
    assert 1 <= shard.is_root.sum() <= 20

    for i in range(len(shard)):
        start, end = shard.row_offsets[i], shard.row_offsets[i + 1]
        probs = shard.policy_prob[start:end]
        idx = shard.policy_index[start:end]
        assert probs.sum() == pytest.approx(1.0, abs=1e-4)
        assert ((idx >= 0) & (idx < 4672)).all()


# --- 5. Training smoke -------------------------------------------------------


def test_training_smoke(tmp_path):
    pytest.importorskip("torch")
    from chessengine.eval.torch_eval import TorchEvaluator
    from chessengine.training.selfplay import run as selfplay_run
    from chessengine.training.train import run as train_run

    net_path = tmp_path / "net.pt"
    TorchEvaluator(blocks=1, filters=8).save(net_path)
    data_dir = tmp_path / "data"
    selfplay_run(
        [
            "--net", str(net_path), "--out", str(data_dir),
            "--games", "2", "--sims", "24", "--workers", "1", "--batch-size", "8",
            "--max-plies", "16", "--snapshot-min-visits", "1",
        ]
    )

    out_path = tmp_path / "candidate.pt"
    losses = train_run(
        [
            "--data", str(data_dir), "--in", str(net_path), "--out", str(out_path),
            "--steps", "20", "--batch", "16", "--min-visits-interior", "1", "--log-every", "10",
        ]
    )
    assert out_path.exists()
    assert math.isfinite(losses["value_loss"])
    assert math.isfinite(losses["policy_loss"])

    reloaded = TorchEvaluator(checkpoint=out_path)
    assert reloaded.model.blocks == 1
    assert reloaded.model.filters == 8


# --- 6. Arena smoke -----------------------------------------------------------


def test_arena_smoke(tmp_path):
    pytest.importorskip("torch")
    from chessengine.eval.torch_eval import TorchEvaluator
    from chessengine.training.arena import run as arena_run

    net_a = tmp_path / "a.pt"
    net_b = tmp_path / "b.pt"
    TorchEvaluator(blocks=1, filters=8).save(net_a)
    TorchEvaluator(blocks=1, filters=8).save(net_b)

    result = arena_run(
        [
            "--net-a", str(net_a), "--net-b", str(net_b),
            "--games", "2", "--sims", "24", "--workers", "1", "--batch-size", "8",
            "--max-plies", "16", "--temp-plies", "2",
        ]
    )
    assert 0.0 <= result["score"] <= 2.0
    assert result["wins"] + result["draws"] + result["losses"] == 2
    # Colors alternate by game index (DESIGN-M6.md section 7.5): the arena
    # driver assigns white via a_plays_white(g), exercised for both games run.
    assert a_plays_white(0) is True
    assert a_plays_white(1) is False


# --- 7. End-to-end mini-generation ------------------------------------------


@pytest.mark.slow
def test_end_to_end_mini_generation(tmp_path, capsys):
    pytest.importorskip("torch")
    from chessengine.eval.torch_eval import TorchEvaluator
    from chessengine.training.arena import main as arena_main
    from chessengine.training.selfplay import main as selfplay_main
    from chessengine.training.train import main as train_main

    # Through the real main() entry points (the console-script surface, not
    # run()): main() returns None by design (the generated console-script
    # wrapper does sys.exit(main()), and sys.exit() prints+exits 1 for any
    # non-None, non-int argument — see selfplay.run()'s docstring). So this
    # test checks main()'s side effects (files on disk, printed summaries)
    # instead of a return value, exactly as a real `chessengine-*` shell
    # invocation would have to.
    best_path = tmp_path / "best.pt"
    assert train_main(["--init", "--out", str(best_path)]) is None
    assert best_path.exists()

    data_dir = tmp_path / "data"
    assert (
        selfplay_main(
            [
                "--net", str(best_path), "--out", str(data_dir),
                "--games", "2", "--sims", "24", "--workers", "1", "--batch-size", "8",
                "--max-plies", "16", "--snapshot-min-visits", "1",
            ]
        )
        is None
    )
    assert len(list(data_dir.glob("game-*.npz"))) == 2

    candidate_path = tmp_path / "candidate.pt"
    capsys.readouterr()  # clear buffered output from the steps above
    assert (
        train_main(
            [
                "--data", str(data_dir), "--in", str(best_path), "--out", str(candidate_path),
                "--steps", "20", "--batch", "16", "--min-visits-interior", "1",
            ]
        )
        is None
    )
    assert candidate_path.exists()
    printed = capsys.readouterr().out
    avg_line = next(line for line in printed.splitlines() if line.startswith("avg "))
    # "avg value_loss <v>  avg policy_loss <p>" -> the numbers are parts[2], parts[5].
    parts = avg_line.split()
    assert math.isfinite(float(parts[2]))
    assert math.isfinite(float(parts[5]))

    capsys.readouterr()
    assert (
        arena_main(
            [
                "--net-a", str(candidate_path), "--net-b", str(best_path),
                "--games", "2", "--sims", "24", "--workers", "1", "--batch-size", "8",
                "--max-plies", "16", "--temp-plies", "2",
            ]
        )
        is None
    )
    score_line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("A score:")
    )
    score = float(score_line.split()[2].split("/")[0])
    assert 0.0 <= score <= 2.0


def test_main_entry_points_return_none(tmp_path):
    """The auto-generated console-script wrapper does sys.exit(main()); a
    non-None return would print itself and force exit code 1. Every main()
    in this package must return None (see selfplay.run()'s docstring)."""
    pytest.importorskip("torch")
    from chessengine.training.train import main as train_main

    assert train_main(["--init", "--out", str(tmp_path / "net.pt")]) is None

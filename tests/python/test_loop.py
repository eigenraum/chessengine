"""Smoke test for the automated generation loop (chessengine-loop)."""

from __future__ import annotations

import pytest

from chessengine.training.loop import decide_candidate_action


@pytest.mark.parametrize(
    "promote, auto_promote, fraction, keep_threshold, expected",
    [
        (True, True, 0.6, 0.5, "promote"),
        (True, False, 0.6, 0.5, "keep"),  # cleared the gate but auto-promote is off
        (False, True, 0.51, 0.5, "keep"),  # missed the gate, still above keep-threshold
        (False, True, 0.5, 0.5, "discard"),  # exactly at keep-threshold: not kept
        (False, True, 0.2, 0.5, "discard"),
    ],
)
def test_decide_candidate_action(promote, auto_promote, fraction, keep_threshold, expected):
    assert decide_candidate_action(promote, auto_promote, fraction, keep_threshold) == expected


def test_loop_smoke_two_generations(tmp_path):
    pytest.importorskip("torch")
    from tensorboard.backend.event_processing import event_accumulator

    from chessengine.training.loop import run as loop_run

    best_path = tmp_path / "best.pt"
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    assert not best_path.exists()

    loop_run(
        [
            "--best", str(best_path), "--data", str(data_dir),
            "--tensorboard-dir", str(runs_dir), "--generations", "2",
            "--selfplay-games", "2", "--selfplay-sims", "16", "--workers", "1",
            "--batch-size", "8", "--parallel-games", "2", "--device", "cpu",
            "--max-plies", "10", "--train-steps", "8", "--train-batch", "8",
            "--min-visits-interior", "1", "--arena-games", "2", "--arena-sims", "16",
            # keep-threshold 0: every candidate survives regardless of arena
            # score, so the file-count assertion below is deterministic
            # (decide_candidate_action's keep/discard split is unit-tested
            # separately, above).
            "--keep-threshold", "0",
        ]
    )

    # best.pt is created (generation-0 bootstrap) and survives every generation.
    assert best_path.exists()
    # Each generation's candidate is kept on disk under its own name (not
    # clobbered), whether or not it cleared the arena gate.
    candidates = sorted(tmp_path.glob("candidate-gen*.pt"))
    assert len(candidates) == 2

    ea = event_accumulator.EventAccumulator(str(runs_dir))
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in (
        "selfplay/games", "train/value_loss", "train/policy_loss",
        "arena/score_fraction", "arena/promoted",
    ):
        assert tag in tags
        assert len(ea.Scalars(tag)) == 2


def test_loop_rejects_jobs_and_parallel_games_together(tmp_path):
    from chessengine.training.loop import run as loop_run

    with pytest.raises(SystemExit):
        loop_run(
            [
                "--best", str(tmp_path / "best.pt"), "--data", str(tmp_path / "data"),
                "--generations", "1", "--jobs", "2", "--parallel-games", "2",
            ]
        )

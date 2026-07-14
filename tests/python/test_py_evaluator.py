"""PyEvaluator bridge tests with fake Python callbacks — no torch needed
(DESIGN-M6.md section 5). test_torch_eval.py covers the real net.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from chessengine import _mcts
from chessengine.engine import Engine, EngineConfig, SearchLimits

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _root_priors(engine: Engine) -> list[tuple[str, float, int]]:
    # (move, prior, visits) for the rows whose parent is row 0, the root.
    view = engine.tree_view()
    return [
        (m, p, v)
        for parent, m, p, v in zip(view.parent, view.move, view.prior, view.visits)
        if parent == 0
    ]


def _uniform_cb(planes: np.ndarray):
    n = planes.shape[0]
    return (
        np.full(n, 0.5, dtype=np.float32),
        np.zeros((n, _mcts.POLICY_SIZE), dtype=np.float32),
    )


def test_uniform_logits_yield_uniform_priors():
    with Engine(EngineConfig(workers=1, evaluator=_uniform_cb)) as engine:
        engine.set_position(STARTPOS)
        engine.search(SearchLimits(max_time_ms=0, max_simulations=200, convergence_window=0))
        priors = [p for _, p, _ in _root_priors(engine)]
        assert len(priors) > 1
        assert priors == pytest.approx([priors[0]] * len(priors), abs=1e-4)
        assert sum(priors) == pytest.approx(1.0, abs=1e-4)


def test_crafted_logits_dominate_prior_and_selection():
    target_idx = _mcts.move_indices(STARTPOS, ["e2e4"])[0]

    def crafted_cb(planes: np.ndarray):
        n = planes.shape[0]
        logits = np.zeros((n, _mcts.POLICY_SIZE), dtype=np.float32)
        logits[:, target_idx] = 10.0
        return np.full(n, 0.5, dtype=np.float32), logits

    with Engine(EngineConfig(workers=1, evaluator=crafted_cb)) as engine:
        engine.set_position(STARTPOS)
        engine.search(SearchLimits(max_time_ms=0, max_simulations=200, convergence_window=0))
        root = _root_priors(engine)
        e2e4_prior = next(p for m, p, _ in root if m == "e2e4")
        assert e2e4_prior == pytest.approx(1.0, abs=1e-3)
        most_visited_move = max(root, key=lambda row: row[2])[0]
        assert most_visited_move == "e2e4"


def test_values_flow_through_single_simulation():
    # A single simulation evaluates exactly one root child; its 0.9 is the
    # win probability for the side to move there (the opponent of the root's
    # side to move), so the root's side-to-move value is 1 - 0.9. Many
    # simulations would make this tree-shape-dependent and flaky — keep it
    # to one.
    def constant_cb(planes: np.ndarray):
        n = planes.shape[0]
        return (
            np.full(n, 0.9, dtype=np.float32),
            np.zeros((n, _mcts.POLICY_SIZE), dtype=np.float32),
        )

    with Engine(EngineConfig(workers=1, evaluator=constant_cb)) as engine:
        engine.set_position(STARTPOS)
        result = engine.search(
            SearchLimits(max_time_ms=0, max_simulations=1, convergence_window=0)
        )
        assert result.root_value == pytest.approx(0.1, abs=1e-3)


def test_liveness_polling_from_main_thread_does_not_deadlock():
    def slow_cb(planes: np.ndarray):
        time.sleep(0.01)
        return _uniform_cb(planes)

    with Engine(EngineConfig(workers=2, batch_size=8, evaluator=slow_cb)) as engine:
        engine.set_position(STARTPOS)
        engine.start(SearchLimits(max_time_ms=2000, max_simulations=-1, convergence_window=0))
        for _ in range(5):
            stats = engine.stats()
            assert stats.simulations >= 0
            time.sleep(0.02)
        result = engine.stop()
        assert result.stop_reason == "interrupted"


def test_broken_callback_falls_back_to_neutral_values():
    def broken_cb(planes: np.ndarray):
        raise RuntimeError("boom")

    with Engine(EngineConfig(workers=1, evaluator=broken_cb)) as engine:
        engine.set_position(STARTPOS)
        result = engine.search(
            SearchLimits(max_time_ms=0, max_simulations=50, convergence_window=0)
        )
        assert result.simulations == 50
        assert result.stop_reason == "simulations"


def test_malformed_return_shape_falls_back_to_neutral_values():
    # Distinct from test_broken_callback_falls_back_to_neutral_values: this
    # callback doesn't raise in Python, it returns garbage. py::cast /
    # array_t::unchecked<N>() reject it with a plain C++ exception (not
    # py::error_already_set) — that path must be survivable too.
    def malformed_cb(planes: np.ndarray):
        n = planes.shape[0]
        return np.full((n, 2), 0.5, dtype=np.float32), "not an array"

    with Engine(EngineConfig(workers=1, evaluator=malformed_cb)) as engine:
        engine.set_position(STARTPOS)
        result = engine.search(
            SearchLimits(max_time_ms=0, max_simulations=50, convergence_window=0)
        )
        assert result.simulations == 50
        assert result.stop_reason == "simulations"


def test_batch_shape_and_dtype():
    seen = []

    def checking_cb(planes: np.ndarray):
        assert planes.shape[1:] == (19, 8, 8)
        assert planes.dtype == np.float32
        assert planes.shape[0] <= 8
        seen.append(planes.shape[0])
        return _uniform_cb(planes)

    with Engine(EngineConfig(workers=1, batch_size=8, evaluator=checking_cb)) as engine:
        engine.set_position(STARTPOS)
        engine.search(SearchLimits(max_time_ms=0, max_simulations=50, convergence_window=0))
    assert seen  # the callback was actually exercised

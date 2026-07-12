"""Parallel search validation: the shared-tree workers must reproduce the
sequential reference behavior (workers=1 is validated in test_search.py)."""

import time

import chess
import pytest

from chessengine import Engine, EngineConfig, SearchLimits


def search(fen: str, workers: int, sims: int):
    engine = Engine(EngineConfig(workers=workers))
    engine.set_position(fen)
    return engine.search(
        SearchLimits(max_time_ms=60_000, max_simulations=sims, convergence_window=0)
    )


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_finds_mate_in_one(workers):
    result = search("k7/8/K7/8/8/8/8/7R w - - 0 1", workers, sims=4000)
    assert result.best_move == "h1h8"
    assert result.root_value > 0.9


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_captures_hanging_queen(workers):
    result = search("k7/8/8/3q4/8/8/8/3RK3 w - - 0 1", workers, sims=8000)
    assert result.best_move == "d1d5"


def test_parallel_simulation_count_is_exact():
    result = search(chess.STARTING_FEN, workers=4, sims=5000)
    assert result.simulations == 5000
    assert result.stop_reason == "simulations"


def test_parallel_value_close_to_sequential():
    fen = "k7/8/8/8/8/8/8/QK6 w - - 0 1"  # white up a queen
    sequential = search(fen, workers=1, sims=4000)
    parallel = search(fen, workers=4, sims=4000)
    assert abs(sequential.root_value - parallel.root_value) < 0.1
    assert parallel.root_value > 0.8


def test_start_stop_interrupts():
    engine = Engine(EngineConfig(workers=4))
    engine.set_position(chess.STARTING_FEN)
    engine.start(SearchLimits(max_time_ms=60_000, convergence_window=0))
    assert engine.running()
    time.sleep(0.2)
    result = engine.stop()
    assert result.stop_reason == "interrupted"
    assert not engine.running()
    assert result.best_move
    assert result.simulations > 0


def test_stats_while_running():
    engine = Engine(EngineConfig(workers=2))
    engine.set_position(chess.STARTING_FEN)
    engine.start(SearchLimits(max_time_ms=60_000, convergence_window=0))
    time.sleep(0.2)
    stats = engine.stats()
    result = engine.stop()
    assert stats.simulations > 0
    assert stats.best_move
    assert result.simulations >= stats.simulations


def test_engine_reusable_after_stop():
    engine = Engine(EngineConfig(workers=2))
    engine.set_position(chess.STARTING_FEN)
    engine.start(SearchLimits(max_time_ms=60_000, convergence_window=0))
    engine.stop()
    engine.set_position("k7/8/K7/8/8/8/8/7R w - - 0 1")
    result = engine.search(
        SearchLimits(max_time_ms=60_000, max_simulations=2000, convergence_window=0)
    )
    assert result.best_move == "h1h8"


def test_stop_after_natural_finish_returns_result():
    engine = Engine(EngineConfig(workers=2))
    engine.set_position(chess.STARTING_FEN)
    engine.start(SearchLimits(max_time_ms=60_000, max_simulations=200, convergence_window=0))
    while engine.running():
        time.sleep(0.01)
    result = engine.stop()
    assert result.stop_reason == "simulations"
    assert result.simulations == 200

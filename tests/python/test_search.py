"""Sequential MCTS behavior: legality, determinism, tactics, termination."""

import chess
import pytest

from chessengine import Engine, EngineConfig, SearchLimits


def make_engine(**config_kwargs) -> Engine:
    return Engine(EngineConfig(**config_kwargs))


def search(fen: str, sims: int = 2000, **limit_kwargs):
    engine = make_engine()
    engine.set_position(fen)
    limits = SearchLimits(
        max_time_ms=60_000, max_simulations=sims, convergence_window=0, **limit_kwargs
    )
    return engine.search(limits)


def test_returns_legal_move_from_startpos():
    result = search(chess.STARTING_FEN, sims=500)
    assert chess.Move.from_uci(result.best_move) in chess.Board().legal_moves
    assert result.stop_reason == "simulations"
    assert result.simulations == 500
    assert result.pv[0] == result.best_move
    assert 0.4 < result.root_value < 0.6  # startpos is roughly balanced


def test_sequential_search_is_deterministic():
    results = [search(chess.STARTING_FEN, sims=1000) for _ in range(2)]
    assert results[0].best_move == results[1].best_move
    assert results[0].root_value == results[1].root_value
    assert results[0].nodes == results[1].nodes
    assert results[0].pv == results[1].pv


def test_finds_mate_in_one():
    # White: Ka6, Rh1 vs Black: Ka8 — only h1h8 mates.
    result = search("k7/8/K7/8/8/8/8/7R w - - 0 1", sims=2000)
    assert result.best_move == "h1h8"
    assert result.root_value > 0.9


def test_captures_hanging_queen():
    # Undefended black queen on d5; white rook on d1 takes it.
    result = search("k7/8/8/3q4/8/8/8/3RK3 w - - 0 1", sims=4000)
    assert result.best_move == "d1d5"


def test_material_eval_reflected_in_root_value():
    # White is up a full queen: root value should clearly favor white.
    result = search("k7/8/8/8/8/8/8/QK6 w - - 0 1", sims=500)
    assert result.root_value > 0.8
    assert result.root_cp > 200


def test_no_legal_moves_stalemate():
    # Black to move, stalemated.
    engine = make_engine()
    engine.set_position("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
    result = engine.search(SearchLimits(max_time_ms=1000))
    assert result.stop_reason == "no_legal_moves"
    assert result.best_move == ""


def test_convergence_stops_early():
    # Mate in 1: the evaluation and best move lock in almost immediately.
    engine = make_engine()
    engine.set_position("k7/8/K7/8/8/8/8/7R w - - 0 1")
    limits = SearchLimits(
        max_time_ms=60_000,
        convergence_window=400,
        convergence_cp_threshold=10,
    )
    result = engine.search(limits)
    assert result.stop_reason == "converged"
    assert result.best_move == "h1h8"


def test_time_limit_respected():
    engine = make_engine()
    engine.set_position(chess.STARTING_FEN)
    result = engine.search(SearchLimits(max_time_ms=200, convergence_window=0))
    assert result.stop_reason == "time"
    assert result.elapsed_ms < 2000  # generous slack for slow CI


def test_stats_after_search():
    engine = make_engine()
    engine.set_position(chess.STARTING_FEN)
    result = engine.search(SearchLimits(max_simulations=300, convergence_window=0))
    stats = engine.stats()
    assert stats.simulations == result.simulations
    assert stats.best_move == result.best_move


def test_larger_batch_size_still_works_sequentially():
    engine = make_engine(batch_size=32)
    engine.set_position(chess.STARTING_FEN)
    result = engine.search(SearchLimits(max_simulations=300, convergence_window=0))
    assert result.simulations == 300
    assert result.best_move

"""Tree reuse across moves: advance() re-roots into the played subtree."""

import chess
import pytest

from chessengine import Engine, EngineConfig, SearchLimits


def limits(sims: int) -> SearchLimits:
    return SearchLimits(max_time_ms=60_000, max_simulations=sims, convergence_window=0)


def test_advance_carries_subtree_over():
    engine = Engine(EngineConfig())
    engine.set_position(chess.STARTING_FEN)
    result = engine.search(limits(5000))
    engine.advance(result.best_move)
    stats = engine.stats()
    # The new root is the old best child: heavily explored, but smaller than
    # the whole old tree.
    assert 1 < stats.nodes < result.nodes
    assert stats.best_move  # a reply is already explored, tree starts warm


def test_advance_unexplored_position_starts_fresh():
    engine = Engine(EngineConfig())
    engine.set_position(chess.STARTING_FEN)
    engine.advance("e2e4")  # nothing searched yet -> nothing to carry over
    assert engine.stats().nodes == 1


def test_advance_illegal_move_raises():
    engine = Engine(EngineConfig())
    engine.set_position(chess.STARTING_FEN)
    with pytest.raises(ValueError):
        engine.advance("e2e5")


def test_advance_while_running_raises():
    engine = Engine(EngineConfig())
    engine.set_position(chess.STARTING_FEN)
    engine.start(SearchLimits(max_time_ms=60_000, convergence_window=0))
    with pytest.raises(RuntimeError):
        engine.advance("e2e4")
    engine.stop()


def test_search_after_advance_finds_tactic():
    # Black to move; after black's reply, white's rook takes the hanging queen.
    engine = Engine(EngineConfig())
    engine.set_position("k7/8/8/3q4/8/8/8/3RK3 b - - 0 1")
    engine.search(limits(2000))
    engine.advance("a8b8")
    result = engine.search(limits(4000))
    assert result.best_move == "d1d5"


def test_selfplay_with_advance_stays_consistent():
    """Engine plays itself via advance(); every move must be legal in the
    python-chess game replayed alongside."""
    engine = Engine(EngineConfig(workers=2))
    board = chess.Board()
    engine.set_position(board.fen())
    for _ in range(30):
        result = engine.search(limits(300))
        if not result.best_move:
            break
        move = chess.Move.from_uci(result.best_move)
        assert move in board.legal_moves, f"illegal {result.best_move} at {board.fen()}"
        board.push(move)
        engine.advance(result.best_move)
        if board.is_game_over(claim_draw=True):
            break

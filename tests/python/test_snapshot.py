"""Training-data export: tree_snapshot() row consistency and filters."""

import chess
import pytest

from chessengine import Engine, EngineConfig, SearchLimits

MATE_IN_ONE = "k7/8/K7/8/8/8/8/7R w - - 0 1"


def searched_engine(fen: str, sims: int = 2000) -> Engine:
    engine = Engine(EngineConfig())
    engine.set_position(fen)
    engine.search(SearchLimits(max_time_ms=60_000, max_simulations=sims, convergence_window=0))
    return engine


def test_snapshot_rows_are_consistent():
    engine = searched_engine(MATE_IN_ONE)
    snap = engine.tree_snapshot()
    assert len(snap) == len(snap.fens) == len(snap.visit_counts) == len(snap.values)
    assert len(snap.moves) == len(snap.child_visits) == len(snap)
    assert snap.fens[0] == MATE_IN_ONE  # row 0 is the root
    assert snap.values[0] > 0.9  # white mates


def test_snapshot_policy_target_prefers_mate():
    engine = searched_engine(MATE_IN_ONE)
    snap = engine.tree_snapshot()
    root_moves, root_visits = snap.moves[0], snap.child_visits[0]
    mate_visits = root_visits[root_moves.index("h1h8")]
    # The mate move must be the clear visit leader. Its share stays well below
    # 1.0 with the material evaluator: sibling moves also win a rook's worth
    # of material, so PUCT keeps exploring them.
    assert mate_visits == max(root_visits)
    assert mate_visits / sum(root_visits) > 0.3


def test_snapshot_fens_parse_and_moves_are_legal():
    engine = searched_engine(chess.STARTING_FEN, sims=1500)
    snap = engine.tree_snapshot(min_visits=5)
    assert len(snap) > 1
    for fen, moves in zip(snap.fens, snap.moves):
        board = chess.Board(fen)
        for uci in moves:
            assert chess.Move.from_uci(uci) in board.legal_moves


def test_snapshot_values_flip_perspective():
    # White up a queen, white to move: the root is great for white, and the
    # children rows (black to move) can never favor black — at best black
    # reaches a draw (several queen moves stalemate immediately, value 0.5).
    engine = searched_engine("k7/8/8/8/8/8/8/QK6 w - - 0 1", sims=2000)
    snap = engine.tree_snapshot(min_visits=50, max_depth=1)  # root + its children only
    assert snap.values[0] > 0.8
    child_values = snap.values[1:]
    assert len(child_values) > 0
    assert all(v <= 0.5 + 1e-5 for v in child_values)
    assert min(child_values) < 0.2  # white's good moves leave black lost


def test_snapshot_min_visits_filters_rows():
    engine = searched_engine(chess.STARTING_FEN, sims=2000)
    assert len(engine.tree_snapshot(min_visits=100)) < len(engine.tree_snapshot(min_visits=1))


def test_snapshot_max_depth_limits_rows():
    engine = searched_engine(chess.STARTING_FEN, sims=2000)
    snap = engine.tree_snapshot(min_visits=1, max_depth=1)
    # exactly the root plus its explored children
    assert len(snap) == 1 + len(snap.moves[0])


def test_snapshot_while_running_raises():
    engine = Engine(EngineConfig())
    engine.set_position(chess.STARTING_FEN)
    engine.start(SearchLimits(max_time_ms=60_000, convergence_window=0))
    with pytest.raises(RuntimeError):
        engine.tree_snapshot()
    engine.stop()

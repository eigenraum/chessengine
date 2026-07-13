"""tree_view(): structure invariants, filtering, and safety while running."""

import chess
import pytest

from chessengine.engine import Engine, EngineConfig, SearchLimits

START_FEN = chess.STARTING_FEN


@pytest.fixture
def searched_engine() -> Engine:
    engine = Engine(EngineConfig(workers=1, seed=1))
    engine.set_position(START_FEN)
    engine.search(SearchLimits(max_time_ms=10_000, max_simulations=2_000))
    return engine


def test_structure_invariants(searched_engine):
    view = searched_engine.tree_view()
    n = len(view)
    assert n > 100
    assert (
        n
        == len(view.move)
        == len(view.visits)
        == len(view.q)
        == len(view.prior)
        == len(view.children_total)
    )
    # row 0 is the root; every parent points at an earlier row
    assert view.parent[0] == -1
    assert view.move[0] == ""
    assert all(0 <= view.parent[i] < i for i in range(1, n))
    # the root saw every simulation; children can't out-visit their parent
    assert view.visits[0] == 2_000
    for i in range(1, n):
        assert 0 < view.visits[i] <= view.visits[view.parent[i]]
    assert all(0.0 <= q <= 1.0 for q in view.q)
    # root has 20 legal children in the start position
    assert view.children_total[0] == 20


def test_moves_are_legal(searched_engine):
    view = searched_engine.tree_view()
    board = chess.Board(START_FEN)
    # depth-1 rows: children of the root must be legal opening moves
    legal = {m.uci() for m in board.legal_moves}
    depth1 = [view.move[i] for i in range(1, len(view)) if view.parent[i] == 0]
    assert depth1 and set(depth1) <= legal


def test_max_nodes_budget_and_best_first_order(searched_engine):
    view = searched_engine.tree_view(max_nodes=50)
    assert len(view) == 50
    # best-first invariant (exact on an idle engine): each emitted node is the
    # most-visited of the frontier, so rows come out in non-increasing order
    assert view.visits == sorted(view.visits, reverse=True)


def test_min_visits_filter(searched_engine):
    view = searched_engine.tree_view(min_visits=10)
    assert all(v >= 10 for v in view.visits[1:])


def test_root_path_descends(searched_engine):
    full = searched_engine.tree_view()
    move = full.move[1]  # a well-visited root child
    sub = searched_engine.tree_view(root_path=[move])
    assert len(sub) > 0
    assert sub.move[0] == ""  # subtree root row has no move
    # the subtree root is exactly that child
    child_row = full.move.index(move)
    assert sub.visits[0] == full.visits[child_row]


def test_root_path_unknown_move_gives_empty_view(searched_engine):
    assert len(searched_engine.tree_view(root_path=["e2e5"])) == 0
    assert len(searched_engine.tree_view(root_path=["e2e4", "e2e4"])) == 0


def test_callable_while_running():
    engine = Engine(EngineConfig(workers=2))
    engine.set_position(START_FEN)
    engine.start(SearchLimits(max_time_ms=1_500, convergence_window=0))
    sizes = []
    while engine.running():
        view = engine.tree_view(max_nodes=10_000)
        assert all(0 <= view.parent[i] < i for i in range(1, len(view)))
        sizes.append(len(view))
    engine.stop()
    assert sizes and sizes[-1] >= sizes[0]  # the live view grows with the tree


def test_fresh_engine_has_bare_root():
    engine = Engine(EngineConfig(workers=1))
    engine.set_position(START_FEN)
    view = engine.tree_view()
    assert len(view) == 1
    assert view.visits == [0]

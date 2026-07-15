"""M6a policy plumbing tests (DESIGN-M6.md section 4)."""

import pytest

from chessengine.engine import Engine, EngineConfig, SearchLimits

PIN_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "8/2k5/8/8/8/3QK3/8/8 w - - 0 1",
]

# Fixed on unmodified pre-M6a code (step 0): with noise off, the M6a refactor
# must reproduce these exactly — uniform priors, same expansion/selection
# order as before the interface change.
EXPECTED = {
    PIN_FENS[0]: ("a2a3", 2000, 44518, 0),
    PIN_FENS[1]: ("d7d6", 2000, 64576, -18),
    PIN_FENS[2]: ("d3d1", 2000, 40274, 816),
}


def _fixed_search(fen: str):
    engine = Engine(EngineConfig(workers=1, seed=42))
    engine.set_position(fen)
    result = engine.search(
        SearchLimits(max_time_ms=0, max_simulations=2000, convergence_window=0)
    )
    return result.best_move, result.simulations, result.nodes, result.root_cp


@pytest.mark.parametrize("fen", PIN_FENS)
def test_sequential_search_pinned(fen):
    # With noise off, the whole M6a refactor must be behaviorally invisible:
    # uniform priors, same order. Never change these numbers during M6a.
    assert _fixed_search(fen) == EXPECTED[fen]

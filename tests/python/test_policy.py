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


def _root_priors(engine: Engine) -> list[float]:
    # Root children are the rows whose parent is row 0, the search root.
    view = engine.tree_view()
    return [p for parent, p in zip(view.parent, view.prior) if parent == 0]


def _search_with_noise(seed: int, root_noise_eps: float = 0.0, sims: int = 300, workers: int = 1):
    engine = Engine(EngineConfig(workers=workers, seed=seed))
    engine.set_position(PIN_FENS[0])
    result = engine.search(
        SearchLimits(
            max_time_ms=0,
            max_simulations=sims,
            convergence_window=0,
            root_noise_eps=root_noise_eps,
        )
    )
    return engine, result


def test_uniform_priors_without_noise():
    engine, _ = _search_with_noise(seed=1)
    priors = _root_priors(engine)
    assert len(priors) > 1
    assert priors == pytest.approx([priors[0]] * len(priors), abs=1e-4)
    assert sum(priors) == pytest.approx(1.0, abs=1e-4)


def test_noise_changes_priors():
    engine, _ = _search_with_noise(seed=1, root_noise_eps=0.25)
    priors = _root_priors(engine)
    assert sum(priors) == pytest.approx(1.0, abs=1e-4)
    assert all(p > 0 for p in priors)
    assert len(set(priors)) > 1


def test_noise_deterministic_per_seed():
    engine_a1, _ = _search_with_noise(seed=7, root_noise_eps=0.25)
    engine_a2, _ = _search_with_noise(seed=7, root_noise_eps=0.25)
    assert _root_priors(engine_a1) == _root_priors(engine_a2)

    engine_b, _ = _search_with_noise(seed=8, root_noise_eps=0.25)
    assert _root_priors(engine_b) != _root_priors(engine_a1)


def test_sequential_determinism_survives_noise():
    _, result1 = _search_with_noise(seed=3, root_noise_eps=0.25, sims=500)
    _, result2 = _search_with_noise(seed=3, root_noise_eps=0.25, sims=500)
    assert result1.best_move == result2.best_move
    assert result1.pv == result2.pv


def test_parallel_smoke_with_noise():
    engine, result = _search_with_noise(seed=5, root_noise_eps=0.25, sims=2000, workers=4)
    assert result.stop_reason == "simulations"
    priors = _root_priors(engine)
    assert sum(priors) == pytest.approx(1.0, abs=1e-4)

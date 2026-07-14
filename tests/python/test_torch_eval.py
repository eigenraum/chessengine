"""PolicyValueNet / TorchEvaluator tests (DESIGN-M6.md section 6).

torch is an optional dependency — skip the whole module if it isn't
installed (`uv sync --group train` to run these).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessengine import _mcts  # noqa: E402
from chessengine.engine import Engine, EngineConfig, SearchLimits  # noqa: E402
from chessengine.eval.torch_eval import PolicyValueNet, TorchEvaluator  # noqa: E402

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_net_forward_shapes_and_ranges():
    net = PolicyValueNet(blocks=2, filters=8)
    net.eval()
    planes = torch.zeros(4, _mcts.PLANES, 8, 8)
    with torch.no_grad():
        values, logits = net(planes)

    assert values.shape == (4,)
    assert logits.shape == (4, _mcts.POLICY_SIZE)
    assert torch.all(values > 0) and torch.all(values < 1)
    assert torch.isfinite(logits).all()


def test_save_load_round_trip_reproduces_outputs(tmp_path):
    evaluator = TorchEvaluator(blocks=2, filters=8)
    planes = np.random.default_rng(0).standard_normal((3, _mcts.PLANES, 8, 8)).astype(np.float32)
    values_before, logits_before = evaluator(planes)

    path = tmp_path / "checkpoint.pt"
    evaluator.save(path)
    loaded = TorchEvaluator(checkpoint=path)
    values_after, logits_after = loaded(planes)

    np.testing.assert_array_equal(values_before, values_after)
    np.testing.assert_array_equal(logits_before, logits_after)


def test_save_load_reconstructs_non_default_architecture(tmp_path):
    evaluator = TorchEvaluator(blocks=3, filters=16)
    path = tmp_path / "checkpoint.pt"
    evaluator.save(path)

    loaded = TorchEvaluator(checkpoint=path)
    assert loaded.model.blocks == 3
    assert loaded.model.filters == 16


def test_engine_plays_with_random_weight_net():
    evaluator = TorchEvaluator(blocks=2, filters=8)
    with Engine(EngineConfig(workers=1, batch_size=16, evaluator=evaluator)) as engine:
        engine.set_position(STARTPOS)
        result = engine.search(
            SearchLimits(max_time_ms=0, max_simulations=200, convergence_window=0)
        )
        assert result.best_move in _mcts.legal_moves(STARTPOS)


def test_default_device_is_cpu():
    """Library callers (and every test above) keep the deterministic CPU
    reference path unless they opt in (DESIGN-GPU.md section 4.2/4.3)."""
    evaluator = TorchEvaluator(blocks=2, filters=8)
    assert evaluator.device.type == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_device_parity(tmp_path, device):
    """Same checkpoint, same inputs, cpu vs. accelerator: values/logits must
    match up to floating point — the accelerator is a speed change, not a
    behavior change (DESIGN-GPU.md section 6)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps not available")

    cpu_evaluator = TorchEvaluator(blocks=2, filters=8, device="cpu")
    checkpoint = tmp_path / "checkpoint.pt"
    cpu_evaluator.save(checkpoint)
    device_evaluator = TorchEvaluator(checkpoint=checkpoint, device=device)

    planes = np.random.default_rng(0).standard_normal((5, _mcts.PLANES, 8, 8)).astype(np.float32)
    values_cpu, logits_cpu = cpu_evaluator(planes)
    values_dev, logits_dev = device_evaluator(planes)

    np.testing.assert_allclose(values_cpu, values_dev, atol=1e-4)
    np.testing.assert_allclose(logits_cpu, logits_dev, atol=1e-3)

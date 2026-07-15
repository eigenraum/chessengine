"""EvalServer demux tests (DESIGN-GPU.md section 5/6): many threads submit
distinct batches concurrently through their clients; each must get back
exactly its own rows, whether or not the server actually coalesces them into
one forward pass. `model.eval()` freezes BatchNorm running stats, so a row's
output is independent of whatever else shares its batch — comparing each
thread's result against a lone direct forward on the same rows is a real
correctness check, not just a shape check. Runs entirely on CPU; no
accelerator required.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessengine import _mcts  # noqa: E402
from chessengine.eval.server import EvalServer  # noqa: E402
from chessengine.eval.torch_eval import TorchEvaluator  # noqa: E402


def _planes(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, _mcts.PLANES, 8, 8)).astype(np.float32)


def test_concurrent_clients_get_their_own_rows_back(tmp_path):
    reference = TorchEvaluator(blocks=1, filters=4, device="cpu")
    checkpoint = tmp_path / "net.pt"
    reference.save(checkpoint)

    # coalesce_ms large enough that the barrier below reliably forces every
    # client's submission into the same forward pass.
    server = EvalServer(
        checkpoint=checkpoint, blocks=1, filters=4, device="cpu",
        max_batch=64, coalesce_ms=20.0,
    )
    n_clients = 6
    inputs = [_planes(i, n=1 + i) for i in range(n_clients)]
    expected = [reference(planes) for planes in inputs]
    results: list[tuple | None] = [None] * n_clients
    barrier = threading.Barrier(n_clients)

    def worker(i: int) -> None:
        client = server.client()
        try:
            barrier.wait()
            results[i] = client(inputs[i])
        finally:
            client.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    server.close()

    for i in range(n_clients):
        result = results[i]
        assert result is not None
        values, logits = result
        exp_values, exp_logits = expected[i]
        np.testing.assert_allclose(values, exp_values, atol=1e-4)
        np.testing.assert_allclose(logits, exp_logits, atol=1e-3)


def test_correct_without_coalescing(tmp_path):
    """coalesce_ms=0: each submission effectively runs alone. The demux path
    must still be correct with no concurrent stragglers to merge."""
    reference = TorchEvaluator(blocks=1, filters=4, device="cpu")
    checkpoint = tmp_path / "net.pt"
    reference.save(checkpoint)

    server = EvalServer(
        checkpoint=checkpoint, blocks=1, filters=4, device="cpu",
        max_batch=64, coalesce_ms=0.0,
    )
    client = server.client()
    try:
        for seed in range(4):
            planes = _planes(seed, n=3)
            values, logits = client(planes)
            exp_values, exp_logits = reference(planes)
            np.testing.assert_allclose(values, exp_values, atol=1e-4)
            np.testing.assert_allclose(logits, exp_logits, atol=1e-3)
    finally:
        client.close()
        server.close()


def test_close_drains_pending_and_stops_thread():
    server = EvalServer(blocks=1, filters=4, device="cpu", coalesce_ms=0.0)
    client = server.client()
    values, logits = client(_planes(0, n=2))
    assert values.shape == (2,)
    assert logits.shape[0] == 2
    client.close()
    server.close()
    assert not server._thread.is_alive()
    server.close()  # idempotent


def test_submit_after_close_raises():
    server = EvalServer(blocks=1, filters=4, device="cpu")
    client = server.client()
    server.close()
    with pytest.raises(RuntimeError):
        client(_planes(0, n=1))


def test_call_after_client_close_raises():
    server = EvalServer(blocks=1, filters=4, device="cpu", coalesce_ms=0.0)
    client = server.client()
    client(_planes(0, n=1))
    client.close()
    with pytest.raises(RuntimeError):
        client(_planes(0, n=1))
    server.close()


def test_broken_model_reports_error_without_killing_server():
    """A batch that raises inside the forward pass must fail only its own
    submitters, not wedge the server thread for later, healthy batches
    (mirrors PyEvaluator's callback-survives-exceptions contract, bindings.cpp)."""
    server = EvalServer(blocks=1, filters=4, device="cpu", coalesce_ms=0.0)
    client = server.client()
    try:
        with pytest.raises(Exception):
            client(_planes(0, n=1)[:, :1, :, :])  # wrong plane count -> conv shape error
        values, _ = client(_planes(0, n=1))  # server thread still alive and healthy
        assert values.shape == (1,)
    finally:
        client.close()
        server.close()

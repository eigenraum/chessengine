"""Cross-engine batching for GPU/MPS search-time inference (DESIGN-GPU.md
section 5, the G2 slice). One model on one device, shared by many engines
running as threads in one process: `engine.search()` releases the GIL
(bindings.cpp), so N games can run concurrently, and each engine's
synchronous evaluator callback can block (releasing the GIL) while a single
server thread coalesces everyone's pending batches into one forward pass.

Batch composition is timing-dependent, so results are no longer
bitwise-reproducible across runs (accepted, DESIGN-GPU.md section 1.5) — the
correctness reference stays 1 worker, CPU, TorchEvaluator, one game at a
time.

Each EvalServer's own background thread is the only thread that ever calls
its model, so one server alone is safe. Two *separate* EvalServer instances
on the same accelerator (e.g. arena's one-per-net setup) are a different
story: each has its own background thread, so their forward passes can run
truly concurrently — and on this codebase's PyTorch/MPS build that measurably
segfaults or deadlocks (two threads, two models, both `.to("mps")`, no
chessengine code involved at all — see docs/dev/GPU-G2.md). `_device_lock`
below serializes actual device work across every EvalServer sharing a device
string, process-wide, so at most one forward pass touches a given
accelerator at a time regardless of how many servers exist. cpu is exempt:
concurrent inference from multiple threads is a normal, well-supported case
there.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from chessengine.eval.device import select_device
from chessengine.eval.torch_eval import FILTERS_DEFAULT, PolicyValueNet

_device_locks: dict[str, threading.Lock] = {}
_device_locks_guard = threading.Lock()


def _lock_for(device: torch.device) -> threading.Lock | None:
    """None for cpu (no known concurrency hazard there); otherwise a
    process-wide lock shared by every EvalServer on that device string, so
    two servers never dispatch to the same accelerator at once."""
    if device.type == "cpu":
        return None
    key = str(device)
    with _device_locks_guard:
        lock = _device_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _device_locks[key] = lock
        return lock


@dataclass
class _Submission:
    planes: np.ndarray
    cv: threading.Condition = field(default_factory=threading.Condition)
    done: bool = False
    values: np.ndarray | None = None
    logits: np.ndarray | None = None
    error: Exception | None = None


class _Client:
    """Callable evaluator bound to one EvalServer.

    Closeable so the server's coalescing wait (`_run`) can tell a finished
    game apart from one still searching: an unclosed client would otherwise
    count toward `_registered` forever, making every future batch wait out
    the full `coalesce_ms` window for a straggler that will never submit
    again (DESIGN-GPU.md section 5.2).
    """

    def __init__(self, server: "EvalServer") -> None:
        self._server = server
        self._closed = False

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("EvalServer client used after close()")
        return self._server._submit(planes)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._server._deregister()

    def __enter__(self) -> "_Client":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class EvalServer:
    """One PolicyValueNet on one device, shared by many engines in-process.

    `client()` returns a callable (and closeable) evaluator suitable for
    `EngineConfig(evaluator=...)`. Each client submission blocks until the
    server thread has run it through the model — possibly coalesced with
    other clients' concurrent submissions into a single larger forward pass.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        blocks: int = 4,
        filters: int = FILTERS_DEFAULT,
        device: str = "auto",
        max_batch: int = 1024,
        coalesce_ms: float = 2.0,
    ) -> None:
        self.device = select_device(device)
        if checkpoint is not None:
            data = torch.load(checkpoint, map_location="cpu")
            self.model = PolicyValueNet(blocks=data["blocks"], filters=data["filters"])
            self.model.load_state_dict(data["state_dict"])
        else:
            self.model = PolicyValueNet(blocks=blocks, filters=filters)
        self.model.eval().to(self.device)
        self._device_lock = _lock_for(self.device)

        self._max_batch = max_batch
        self._coalesce_s = coalesce_ms / 1000.0
        self._cv = threading.Condition()
        self._pending: list[_Submission] = []
        self._registered = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def client(self) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        with self._cv:
            self._registered += 1
        return _Client(self)

    def close(self) -> None:
        """Stop the server thread. Drains whatever is already pending first
        (each submitter still gets an answer); a submission that arrives
        after close() raises in the submitting thread rather than hanging.
        Safe to call more than once."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()
        self._thread.join()

    # ---- internals ---------------------------------------------------

    def _deregister(self) -> None:
        with self._cv:
            self._registered -= 1
            self._cv.notify_all()

    def _submit(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        submission = _Submission(planes)
        with self._cv:
            if self._closed:
                raise RuntimeError("EvalServer is closed")
            self._pending.append(submission)
            self._cv.notify_all()
        with submission.cv:
            while not submission.done:
                submission.cv.wait()
        if submission.error is not None:
            raise submission.error
        return submission.values, submission.logits  # type: ignore[return-value]

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending or self._closed)
                if self._closed and not self._pending:
                    return
                # Coalesce: wait for other registered clients to arrive, up
                # to coalesce_ms — never longer, so a search that is between
                # batches doesn't stall the whole server.
                deadline = time.monotonic() + self._coalesce_s
                while (
                    not self._closed
                    and len(self._pending) < min(self._registered, self._max_batch)
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
                batch = self._pending[: self._max_batch]
                self._pending = self._pending[len(batch) :]
            self._evaluate(batch)

    def _evaluate(self, batch: list[_Submission]) -> None:
        try:
            planes = np.concatenate([s.planes for s in batch], axis=0)
            if self._device_lock is not None:
                with self._device_lock:
                    values, logits = self._forward(planes)
            else:
                values, logits = self._forward(planes)
        except Exception as exc:  # a broken model must not take down the server thread
            for s in batch:
                s.error = exc
        else:
            offset = 0
            for s in batch:
                n = s.planes.shape[0]
                s.values = values[offset : offset + n]
                s.logits = logits[offset : offset + n]
                offset += n
        finally:
            for s in batch:
                with s.cv:
                    s.done = True
                    s.cv.notify_all()

    def _forward(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            x = torch.from_numpy(planes).to(self.device, non_blocking=True)
            values, logits = self.model(x)
        return values.cpu().numpy().astype(np.float32), logits.cpu().numpy().astype(np.float32)

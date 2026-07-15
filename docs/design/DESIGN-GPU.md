# Design Document: Device-Accelerated Evaluation (CUDA / MPS)

Status: **G1 and G2 implemented (2026-07-15)**. Companion to `DESIGN.md` and
`DESIGN-M6.md`. See `docs/dev/GPU-G1.md` and `docs/dev/GPU-G2.md` for what
actually shipped, benchmark results, and gotchas.
Goal: run `PolicyValueNet` inference on CUDA, then MPS, then CPU only as the
last resort — everywhere the net is used at *search* time (`chessengine-selfplay`,
`chessengine-arena`, and the interactive UIs), not just in the offline
training loop, which already auto-selects a device.

Guiding rule as ever: educational project — readability and simplicity win
over the last percent of runtime performance.

---

## 1. Decisions Taken

1. **No C++ changes for the core of this work.** The existing evaluator
   boundary (one synchronous Python callback per batch, on the engine's
   single evaluator thread — DESIGN.md §5, DESIGN-M6.md §5) already carries
   fixed-shape float32 arrays. Which device the forward pass runs on is
   invisible to C++; the acceleration happens entirely inside the callback.
2. **Two shippable slices.** G1 makes the evaluator device-aware (a
   per-batch `to(device)` round-trip) — small, safe, works today with
   `--jobs`-style process parallelism. G2 adds *cross-engine batching*: many
   games in one process, one shared inference server thread that coalesces
   their leaf batches into large GPU batches. G1 is useful on its own and is
   the fallback if G2's coalescing turns out not to pay off.
3. **Device selection is shared, not duplicated.** The `auto → cuda → mps →
   cpu` logic currently private to `train.py` moves to
   `chessengine/eval/device.py` and is reused by train, self-play, arena, and
   the UIs. Same flag everywhere: `--device auto|cpu|cuda|mps` (default
   `auto`).
4. **Process parallelism (`--jobs`) stays and remains the CPU story.** G2's
   thread-based game parallelism (`--parallel-games`) is the GPU story. They
   are separate flags with a documented "pick one" rule, not a combined
   scheduler (deferred, §8).
5. **Bitwise reproducibility is downgraded to per-config reproducibility,
   documented.** GPU kernels and coalesced batch composition change floating
   point results in the last ulps; seeds still make move *sampling*
   deterministic given identical evaluations, but cross-device and
   cross-`--parallel-games` runs will not be move-identical. The correctness
   reference remains: 1 search worker, CPU evaluator, single game at a time.

---

## 2. Why Search-Time Inference Is CPU-Only Today

The offline training loop (`train.py`) owns its batches directly and just
moves them `.to(device)` — that is why it was trivially accelerated. At
search time the path is different:

```
search workers ──(EvalRequest)──► EvalQueue (C++ evaluator thread)
                                        │ evaluate(batch)
                                        ▼
                              PyEvaluator (bindings.cpp)
                                encode planes (no GIL)
                                acquire GIL ── callback(planes [N,19,8,8])
                                        │            │
                                        ▼            ▼
                              write values/priors   TorchEvaluator.__call__
                              back to workers        (CPU, 1 torch thread)
```

Three deliberate constraints created the CPU-only status quo:

| Constraint | Where | Why it exists |
|---|---|---|
| `torch.set_num_threads(1)` | `TorchEvaluator.__init__` | many self-play worker *processes* must not fight over cores |
| tensors never leave CPU | `TorchEvaluator.__call__` | simplest correct thing for M6c |
| parallelism = OS processes (`--jobs`) | `selfplay.py` | sidesteps the GIL entirely; each process has its own engine + evaluator |

None of these is architectural. The callback is synchronous and blocking by
design (workers park on virtual loss, DESIGN.md §4.3), but *what the callback
does inside* — including a device round-trip or handing the batch to a shared
server thread and sleeping — is free.

The one real architectural mismatch is **batch size**: one engine produces
batches of ≤ `batch_size` (default 64) driven by ≤ `workers` (default 2)
in-flight simulations, and early in each search the queue drains batches of
1–4. GPUs only pay for themselves on large batches. G1 accepts this (and
mitigates by raising `--workers`/`--batch-size`); G2 fixes it by merging
batches across concurrent games.

---

## 3. Shared Device Selection (`python/chessengine/eval/device.py`)

New module, torch-importing (so lazily imported, same rule as
`torch_eval.py` — the package must import without the `train` group):

```python
def select_device(requested: str = "auto") -> torch.device
def describe_device(device: torch.device) -> str
```

Bodies move verbatim from `train.py:_select_device/_describe_device`;
`train.py` becomes a caller. `"auto"` resolves cuda → mps → cpu; an explicit
request is honored without fallback (failing loudly beats silently training
on CPU for a week).

CLI surface, identical wording in all four entry points (train already has
it):

```
--device auto|cpu|cuda|mps    (default: auto)
    auto picks cuda, then Apple Silicon (mps), then cpu
```

Each entry point logs the resolved device once at startup via
`describe_device`, as train does today.

---

## 4. G1 — Device-Aware `TorchEvaluator`

### 4.1 The change

`TorchEvaluator` grows a `device` parameter (string, resolved via
`select_device`):

```python
class TorchEvaluator:
    def __init__(self, checkpoint=None, blocks=4, filters=64,
                 device: str = "cpu") -> None:   # "cpu" default: see §4.3
        self.device = select_device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        ...
        self.model.eval().to(self.device)

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            x = torch.from_numpy(planes).to(self.device, non_blocking=True)
            values, logits = self.model(x)
        return (values.cpu().numpy().astype(np.float32),
                logits.cpu().numpy().astype(np.float32))
```

Notes:

- `torch.set_num_threads(1)` becomes conditional: it exists to keep N
  self-play *processes* from oversubscribing cores, which only applies to the
  CPU path. On cuda/mps the CPU does little (encode + copies) and 1 thread is
  what it effectively gets anyway, so the condition is about intent, not
  effect.
- Checkpoint loading keeps `map_location="cpu"` then `.to(device)` — one code
  path regardless of where the checkpoint was written.
- `torch.inference_mode()` replaces `no_grad()` (strictly better here).
- The GIL is held across `__call__` (PyEvaluator acquires it for the whole
  callback). That is unchanged and fine: per process, the only other Python
  thread is the main thread, which is parked inside `engine.search()` with
  the GIL released. Large torch ops release the GIL internally anyway.

### 4.2 Wiring

- `selfplay.py`: `--device` flag → `SelfPlayConfig.device` →
  `_init_worker` → `TorchEvaluator(checkpoint=..., device=...)`. Works
  unchanged under `--jobs` (spawn): each worker process resolves and owns its
  own device context.
- `arena.py`: `--device` flag → both `TorchEvaluator`s.
- `ui/cli.py`, `ui/web/server.py`: same parameter where they construct
  `TorchEvaluator` — playing against the net gets the speedup for free.

### 4.3 `--jobs` × GPU interaction

Multiple processes sharing one GPU works on both CUDA and MPS but is not
free: each CUDA process pays a few hundred MB of context, and kernels from
different processes serialize. The rule of thumb we document (in `--help` and
README):

- **CPU**: `--jobs N` (as today), device `cpu`.
- **GPU, G1 only**: modest `--jobs` (2–4) can still help hide the per-batch
  round-trip latency; beyond that, contention wins.
- **GPU, once G2 lands**: `--jobs 1 --parallel-games N`.

This is also why the `TorchEvaluator` *default* stays `"cpu"` and only the
CLIs pass `--device` (default `auto`) down: library users and tests keep the
deterministic CPU behavior unless someone explicitly opts in at the entry
point.

### 4.4 Expected effect — measure before G2

For the current net (4 blocks × 64 filters) at batch ≤ 64, a GPU forward is
fast but the fixed per-batch cost (GIL acquire, H2D/D2H copies, kernel
launch — especially on MPS, where dispatch overhead is high) is a large
fraction of the total. G1 may be a modest win or even a wash at defaults; it
becomes a clear win with bigger nets and bigger batches (`--workers 8
--batch-size 256` style configs).

So G1 ships with a micro-benchmark (`tests/python/bench_eval.py` or a
`--bench` mode is overkill — a short script under `tools/` is enough):
time `TorchEvaluator.__call__` across batch sizes {1, 8, 64, 256} ×
available devices, plus `games/s` from a 4-game self-play run per device.
The numbers go into the PR description and decide how urgent G2 is.

---

## 5. G2 — Cross-Engine Batching (`EvalServer`)

The GPU-shaped fix for the small-batch problem: run many games *in one
process*, funnel all their leaf batches into one model on one device.

### 5.1 Key insight: no C++ changes needed

`engine.search()` releases the GIL (`bindings.cpp`), so N games can run on N
Python threads concurrently — the C++ searches are truly parallel. Each
engine's evaluator thread calls its Python callback under the GIL, but a
callback may *block on a `threading.Event`* (releasing the GIL) while a
separate server thread coalesces requests from all engines and runs one
forward pass. The existing `PyEvaluator`/`EvalQueue` machinery neither knows
nor cares that its "evaluation" was computed jointly with other engines'.

```
game thread 1 ── Engine 1 ── evaluator thread ──┐ callback: submit + wait
game thread 2 ── Engine 2 ── evaluator thread ──┤►  EvalServer thread
   ...                                          │   coalesce → one forward
game thread N ── Engine N ── evaluator thread ──┘   on cuda/mps → scatter
```

### 5.2 `EvalServer` (`python/chessengine/eval/server.py`)

```python
class EvalServer:
    """One model on one device, shared by many engines in-process.

    Coalesces the per-engine batches arriving via client callbacks into
    single forward passes. Thread-safe; owns a daemon server thread.
    """
    def __init__(self, checkpoint, device="auto",
                 max_batch=1024, coalesce_ms=2.0): ...
    def client(self) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """A callback suitable for EngineConfig(evaluator=...). Registers the
        client; the returned callable submits a batch and blocks until its
        results are ready."""
    def close(self) -> None: ...
```

Mechanics:

- **Submit**: the client callable appends `(planes, event, out_slot)` to a
  locked list, then `event.wait()` — the wait releases the GIL, which is what
  lets other engines' evaluator threads get in.
- **Coalesce**: the server thread wakes on submission and drains what is
  pending, but if fewer clients than `registered` have submitted it waits up
  to `coalesce_ms` for stragglers before running the forward (never longer —
  a search that is between batches must not stall everyone). `np.concatenate`
  the planes, one forward, split values/logits by the recorded row ranges,
  set each event.
- **Shutdown / stragglers**: clients deregister on engine close (the
  `Engine` context manager exit); `close()` fails any still-waiting client
  loudly rather than hanging. The coalesce wait must key on *currently
  searching* clients, not just registered ones — a finished game whose engine
  is still open must not add `coalesce_ms` to every batch. Simplest correct
  version: track registered clients only and keep `coalesce_ms` small; refine
  only if profiling says so.

The batch-composition nondeterminism this introduces is accepted per §1.5.

### 5.3 Self-play driver changes

New flag `--parallel-games N` (default 1). With N > 1:

- `run()` creates one `EvalServer`, then a `ThreadPoolExecutor(N)` where each
  task is `play_one_game(server.client(), ...)` — `play_one_game` already
  takes the evaluator as a parameter, so it is reused unchanged.
- Mutually exclusive with `--jobs > 1` (hard error: the flags answer the same
  question for different devices, §1.4).
- The tqdm bar ticks per completed game exactly as the process-pool path
  does.

### 5.4 Arena changes

Arena games are sequential today; `--parallel-games` parallelizes them the
same way (each game constructs its two engines from two shared `EvalServer`s,
one per net — a server holds one model). Note the two servers halve the
effective per-net batch size; with only one GPU this is still strictly better
than G1. Game scheduling, seeding (`seed + g`), and the W/D/L accounting are
unchanged; only the loop body moves into the executor task.

---

## 6. Testing

- **Device parity** (`test_torch_eval.py`): same random planes through the
  same checkpoint on CPU vs. each available accelerator; values within
  `atol=1e-4`, argmax of legal-move logits identical for a handful of
  positions. Skipped where no accelerator exists (CI stays green on CPU
  runners).
- **`EvalServer` demux** (new `test_eval_server.py`, CPU, no accelerator
  needed): N threads submit distinct batches concurrently through their
  clients against a stub model that encodes the input in its output (e.g.
  value = mean of the batch row); assert every thread gets exactly its own
  rows back, under coalescing forced both on (barrier before submit) and off
  (`coalesce_ms=0`).
- **End-to-end smoke** (`test_training.py`): `selfplay.run()` with
  `--parallel-games 2 --device cpu --games 2`, and the arena equivalent —
  the whole G2 path minus the accelerator, runnable everywhere.
- **Existing suites** are the regression net: `--device cpu --jobs 1`
  self-play must produce byte-identical shards to pre-G1 given the same seed.

---

## 7. Milestones (shippable slices)

### G1 — device-aware evaluator — shipped

`eval/device.py` (extracted from train.py) · `TorchEvaluator(device=)` ·
`--device` in selfplay/arena/cli/web · docs + rule-of-thumb for `--jobs`×GPU
· bench script · device-parity test. Engine behavior at defaults: unchanged.

`tools/bench_eval.py` on an Apple-Silicon MPS device confirmed the small-
batch ceiling this design predicted: cpu ≈ 2,000 positions/s at batch 64;
mps ≈ 550 positions/s at batch 1 (dispatch overhead loses to cpu) but
≈ 40,000–63,000 positions/s at batch 64–256. That gap is what motivated
shipping G2 immediately rather than waiting — see `docs/dev/GPU-G1.md`.

### G2 — in-process parallel games + `EvalServer` — shipped

`eval/server.py` · `--parallel-games` in selfplay and arena (exclusive with
`--jobs`) · demux + smoke tests. Details, test strategy, and gotchas in
`docs/dev/GPU-G2.md`.

---

## 8. Open Points (deliberately deferred)

- **`--jobs` × `--parallel-games` combined** (processes × threads): adds a
  scheduling story for marginal gain on one-GPU boxes. Revisit for multi-GPU.
- **Cross-process inference server** (shared-memory IPC to one GPU owner
  process): strictly more machinery than G2 for the same batches; only
  interesting if the GIL ever becomes the G2 bottleneck. Measure first.
- **fp16 / autocast on CUDA, `torch.compile`**: easy add inside
  `TorchEvaluator`/`EvalServer` later; needs the G1 benchmark as baseline and
  a parity-tolerance decision.
- **Pinned-memory staging buffers**: only worth it once profiles show H2D
  copies dominating (unlikely at 19×8×8 floats).
- **Interactive UIs and G2**: cli/web get G1 only; a single interactive
  search has no sibling games to batch with. Their path to GPU efficiency
  would be a bigger `--batch-size`, which they already expose via config.

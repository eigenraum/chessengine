# Dev Notes: G2 — Cross-Engine Batching (`EvalServer`)

Implements `docs/design/DESIGN-GPU.md` §5 (the G2 slice). Written after the
fact — "what actually happened," not design rationale (that's in the design
doc). Read `docs/dev/GPU-G1.md` first; this builds directly on it and on the
benchmark numbers recorded there.

## Why this shipped right after G1, not "if the benchmark says so"

The design doc gated G2 on G1's benchmark. It said so: MPS loses badly at
batch 1 (~550 positions/s, worse than CPU) and wins by ~20-30x at batch
64-256 (~40k-63k positions/s). A single engine's `--workers 2 --batch-size
64` rarely fills a batch of 64 *early* in a search — the queue drains
whatever's accumulated, which starts at 1-4 and grows as the tree widens.
That's exactly the regime G1 alone can't fix, and exactly what G2 is for:
pool several games' leaf batches into one that's actually GPU-sized.

## What changed

- `python/chessengine/eval/server.py` (new): `EvalServer` — one
  `PolicyValueNet` on one device, a background thread, and `client()`
  returning a closeable callable evaluator. See the module docstring and
  DESIGN-GPU.md §5 for the mechanics; the short version: `engine.search()`
  releases the GIL (bindings.cpp), so N games run as N Python threads
  genuinely concurrently, and each one's evaluator callback blocks on a
  `threading.Condition` (which releases the GIL) while the server thread
  drains everyone's pending submissions into one `torch` forward pass.
- `--parallel-games N` (default 1) added to `chessengine-selfplay` and
  `chessengine-arena`. Selfplay: `--jobs > 1` together with
  `--parallel-games > 1` is a hard `parser.error` — they answer the same
  "how do I parallelize" question for different hardware (processes for
  CPU, threads sharing a server for GPU). Arena never had `--jobs`, so no
  conflict there — it just gained the flag.
- Arena's `--parallel-games` path uses **per-game seeded rng**
  (`np.random.default_rng(seed + game_index)`) instead of the one rng
  threaded through the whole sequential run. A single shared
  `np.random.Generator` isn't safe to call concurrently from multiple
  threads; per-game seeding sidesteps that and matches what selfplay already
  does (`game_seed = seed + game_index`). This only affects the new parallel
  path — the sequential arena loop is untouched.
- `tests/python/test_eval_server.py` (new): pure-CPU demux tests. No GPU
  needed to test the coalescing logic itself, only to benefit from it.
- Smoke tests added to `test_training.py`:
  `test_selfplay_parallel_games_smoke`, `test_selfplay_parallel_games_rejects_jobs`,
  `test_arena_parallel_games_smoke` — all `--device cpu`, exercising the
  real driver code paths end to end.

No C++ changes, same as G1.

## Bug found after shipping: two EvalServers on one accelerator hang/crash

Found via a real user run: `chessengine-arena ... --device mps
--parallel-games 8` hung indefinitely (`arena: 0%|...| 0/N`, no progress,
no error). Arena is the *only* place two `EvalServer`s exist in the same
process at once — one per net — each with its own background thread. Bare,
minimal repro with no chessengine code at all confirmed it's a PyTorch/MPS
issue, not ours:

```python
import threading, torch
from torch import nn

def make_model():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).eval().to("mps")

model_a, model_b = make_model(), make_model()

def worker(model):
    for _ in range(5):
        with torch.inference_mode():
            model(torch.randn(8, 64, device="mps")).cpu()

t1 = threading.Thread(target=worker, args=(model_a,))
t2 = threading.Thread(target=worker, args=(model_b,))
t1.start(); t2.start(); t1.join(); t2.join()
```

Two threads, two independent models, both `.to("mps")`, run concurrently:
**segfault** (exit code 139) on one run, a plain **hang** on another —
nondeterministic, but always broken. A follow-up test showed one `EvalServer`
with *multiple client threads* (the selfplay case — one model, one
background thread, N game-threads submitting to it) is completely fine; the
hazard is specifically **two separate threads both dispatching forward
passes to the same accelerator concurrently**, regardless of which Python
object structure gets them there.

**Fix**: `_lock_for(device)` in `server.py` — a process-wide
`dict[str, threading.Lock]` keyed by device string
(`_device_locks`/`_device_locks_guard`), handed out once per device and
shared by every `EvalServer` on that device. `EvalServer._evaluate` acquires
its device's lock around the actual forward pass (factored into
`_forward()`), so at most one thread ever touches a given accelerator at a
time, no matter how many `EvalServer`s exist. `cpu` is exempt — concurrent
inference from multiple threads there is normal and well-supported; only
non-cpu devices get a lock.

This does mean arena's two nets (`--parallel-games` with `--device mps`)
now take turns on the device rather than truly overlapping — that's the
right trade for correctness over the small amount of extra overlap it costs,
and matches the design doc's honest expectation (§5.4: "two servers halve
the effective per-net batch size") anyway, just enforced with a lock instead
of assumed free.

Regression test: `test_concurrent_servers_on_same_accelerator_do_not_hang`
in `test_eval_server.py`, parametrized over `cuda`/`mps`, skips when the
device isn't present. Bounded with a `join(timeout=30)` so a *hang*
regression fails an assertion instead of freezing CI — but note a *segfault*
regression (the other failure mode actually observed) kills the whole test
process regardless of any Python-level timeout; accepted, since this test
only runs on machines with an actual accelerator.

## The key correctness trick used in testing

`model.eval()` freezes BatchNorm running statistics, so a row's output
depends only on that row's own input — never on which other rows happen to
share its batch. That means the demux test doesn't need a specially crafted
stub model: run the same tiny net once per row alone (the "expected" value)
and once through N concurrent `EvalServer` clients (forced to coalesce via a
`threading.Barrier`), and the two must match up to floating point. This is a
much stronger check than "shapes line up" — it actually proves client A
never receives client B's rows.

## Coalescing implementation notes (`EvalServer._run`)

- Bookkeeping (append to `_pending`, decide batch composition) happens under
  one `threading.Condition`'s lock; the actual `model.forward` call happens
  **outside** the lock, so new submissions from other threads can queue up
  for the *next* batch while the current one runs.
- The wait-for-stragglers loop is bounded by `coalesce_ms` (default 2ms) and
  also exits early once `len(pending) >= min(registered, max_batch)` — i.e.
  once every currently-registered client has something in flight, there's no
  reason to wait further.
- `_Client.close()` (called by both selfplay's and arena's per-game workers
  in a `finally` block) decrements `_registered`. Without this, a finished
  game's client would count toward `_registered` forever, and every later
  batch would wait out the full `coalesce_ms` for a straggler that will
  never submit again. This was in the design doc's "shutdown/stragglers"
  bullet (§5.2) — implemented pretty much as written there.
- A broken model (bad shapes, NaNs that raise, whatever) fails only the
  submissions in the batch that triggered it — each waiting thread re-raises
  the exception locally — and the server thread survives to serve the next
  batch. Mirrors `PyEvaluator`'s "a broken callback must not take down the
  search thread" contract in `bindings.cpp`; tested in
  `test_broken_model_reports_error_without_killing_server`.
- `close()` drains whatever's already pending before stopping the thread (no
  submitter is left hanging), and a submission attempted after `close()`
  raises immediately rather than blocking forever.

## Verifying this slice

```sh
uv run pytest tests/python/test_eval_server.py -v      # pure-CPU demux tests
uv run pytest tests/python/test_training.py -v -k parallel_games
uv run pytest -q                                        # full suite
```

Manual end-to-end check on real hardware, comparing `--jobs` vs.
`--parallel-games` throughput for a representative net size, is the next
useful data point (the microbenchmark in `tools/bench_eval.py` measures raw
forward-pass throughput, not games/s under real self-play — those can differ
because of encode/copy overhead, search-side bottlenecks, and coalescing
wait time).

## Gotchas hit while implementing

- Forgot at first that a shared `np.random.Generator` can't be called from
  multiple threads safely — arena's sequential path threads one `rng` object
  through every game precisely because it *is* sequential. The parallel path
  needed its own per-game rng from the start; don't copy the sequential
  pattern there.
- `EvalServer`'s constructor mirrors `TorchEvaluator`'s
  (`checkpoint`/`blocks`/`filters`/`device`) rather than taking a
  pre-built model, so production and test code share the same construction
  path — no separate "test-only" API surface.
- `_FILTERS_DEFAULT` in `torch_eval.py` had to be renamed to `FILTERS_DEFAULT`
  (drop the leading underscore) once `server.py` needed to import it — it's
  no longer private to one module.

## Next

Nothing else planned in DESIGN-GPU.md beyond what's in §8 (Open Points):
`--jobs` × `--parallel-games` combined scheduling, cross-process inference
server, fp16/`torch.compile`, pinned memory. All explicitly deferred pending
evidence they're needed — don't build any of them speculatively.

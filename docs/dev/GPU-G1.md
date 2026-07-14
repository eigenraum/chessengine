# Dev Notes: G1 — Device-Aware Evaluator

Implements `docs/design/DESIGN-GPU.md` §4 (the G1 slice). Written after the
fact, for whoever (human or Claude) picks up G2 next — not a design
rationale (that's DESIGN-GPU.md), a "what actually happened" record.

## What changed

- `python/chessengine/eval/device.py` (new): `select_device`/`describe_device`/
  `DEVICE_CHOICES`, lifted verbatim out of `train.py`'s private
  `_select_device`/`_describe_device`. Both lazy-import torch inside the
  function body, so importing this module doesn't pull in torch — same rule
  `torch_eval.py` already followed.
- `TorchEvaluator.__init__` takes `device: str = "cpu"` (default stays CPU —
  see "Why the default is cpu, not auto" below), resolves it via
  `select_device`, moves the model there. `set_num_threads(1)` is now
  conditional on `device.type == "cpu"` (it exists to stop N self-play
  *processes* fighting over cores; irrelevant once the forward pass isn't on
  the CPU).
- `TorchEvaluator.__call__` does the `.to(device)` round-trip and uses
  `torch.inference_mode()` instead of `no_grad()`.
- `--device auto|cpu|cuda|mps` added to `chessengine-selfplay`,
  `chessengine-arena`, `chessengine` (CLI), `chessengine-web`. All default to
  `cpu`. `train.py` already had the flag; it now imports the shared helpers
  instead of defining its own.
- `tools/bench_eval.py` (new): times `TorchEvaluator.__call__` across
  devices × batch sizes. Not a test — run it by hand when the question "is
  G2 worth it" comes up again.
- Tests: `test_default_device_is_cpu` (no accelerator needed) and
  `test_device_parity[cuda|mps]` (skips if the device isn't present) in
  `tests/python/test_torch_eval.py`.

No C++ changes. No change to `EvalQueue`/`PyEvaluator`/the search — this was
entirely inside the Python callback, as the design doc predicted.

## Why the default is `cpu`, not `auto`

`TorchEvaluator()` with no arguments is what the test suite and any library
caller gets. Keeping that CPU means the existing determinism guarantees
(byte-identical shards for a given seed) don't shift under people who haven't
opted in. Only the CLI entry points expose `--device` (default `cpu` there
too, explicit opt-in via the flag) — `train.py` is the one exception, already
defaulting its flag to `auto` before this change, left as-is.

## Verifying this slice

```sh
uv sync --group train                       # torch isn't in the default install
uv run pytest tests/python/test_torch_eval.py -v   # device_parity should PASS on mps (Apple
                                                    # Silicon) or SKIP; cuda SKIPs without a GPU
uv run pytest -q                            # full suite, should be unaffected
uv run python tools/bench_eval.py           # prints ms/call and positions/s per device × batch
```

## Benchmark result (recorded here so G2's "is it worth it" question has a
data point; re-run on your own hardware before trusting it)

Apple M-series, `blocks=4, filters=64` (the checkpoint default):

| device | batch | positions/s |
|---|---|---|
| cpu | 64  | ~2,000 |
| mps | 1   | ~550 (dispatch overhead dominates — worse than cpu) |
| mps | 64  | ~40,000 |
| mps | 256 | ~63,000 |

Confirms DESIGN-GPU.md §4.4's prediction: MPS loses at the batch sizes a
*single* engine naturally produces early in a search, and wins by ~20x once
the batch is large. This is the actual argument for G2 — one engine alone
won't reliably produce batch-256 without cranking `--workers`/`--batch-size`
well past their current defaults (2 / 64).

## Gotchas hit while implementing

- `torch` is an optional dependency group (`uv sync --group train`); forgot
  this once and the whole `test_torch_eval.py` module silently skipped
  (`pytest.importorskip("torch")` at module level) instead of failing loudly.
  If a torch test appears to "pass" as 0 collected, check the group is
  installed.
- Pyright flags `Batch.planes: object` (`.to()` unknown on `object`) in
  `train.py` — pre-existing, `dataset.py` deliberately types tensor fields as
  `object` to avoid a module-level torch import. Not introduced by this
  change; don't try to "fix" it by importing torch at module level there.
- `torch.inference_mode()` > `torch.no_grad()` for pure inference (slightly
  more restrictive, and that's fine here — nothing downstream needs
  autograd-compatible tensors out of the evaluator).

## Next: G2

Gate on whether real self-play throughput (not the microbenchmark) actually
improves with `--device mps --workers 8 --batch-size 256` compared to
`--device cpu --jobs <cores>` on a representative net size. If the per-engine
batch can be pushed close to the 256 row by just raising `--workers`/
`--batch-size`, G2 (`EvalServer`, DESIGN-GPU.md §5) may not be worth its
complexity — measure before building it.

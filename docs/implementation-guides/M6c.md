# Implementation Guide: M6c — Self-Play, Training, Arena

Implements `docs/design/DESIGN-M6.md` §7. **Prerequisites: M6a and M6b are
merged** (root noise, `TorchEvaluator`, `_mcts.encode_planes`,
`_mcts.move_indices`, `EngineConfig.evaluator`, `Engine.close()`).

This slice is pure Python. After it, one full generation runs:
self-play → train → arena → promote.

Ground rules:

- Test: `uv run pytest` (torch tests need `uv sync --group train`).
- torch imports stay inside the training modules / lazy paths.
- All knobs are CLI flags with the defaults given here (DESIGN.md
  principle: every performance-relevant knob is configurable).

Files:

```
python/chessengine/training/__init__.py   # empty
python/chessengine/training/selfplay.py
python/chessengine/training/dataset.py
python/chessengine/training/train.py
python/chessengine/training/arena.py
pyproject.toml                            # three console scripts
tests/python/test_training.py             # new
docs/design/DESIGN-M6.md                  # tick off progress if you like; no edits required
```

`pyproject.toml` `[project.scripts]` additions:

```toml
chessengine-selfplay = "chessengine.training.selfplay:main"
chessengine-train = "chessengine.training.train:main"
chessengine-arena = "chessengine.training.arena:main"
```

---

## Step 1 — Shard format (module `dataset.py`, used by `selfplay.py`)

One `.npz` per game, `game-<timestamp>-<pid>-<n>.npz`, fields (flat arrays
over all exported rows of all per-move snapshots, DESIGN-M6.md §7.2):

| field | dtype | content |
|---|---|---|
| `fens` | object (str) | position per row |
| `policy_index` | int32 | ragged: move indices (via `_mcts.move_indices`) |
| `policy_prob` | float32 | ragged: normalized child visits, aligned with `policy_index` |
| `row_offsets` | int64, len n+1 | row i owns `policy_index[row_offsets[i]:row_offsets[i+1]]` |
| `search_value` | float32 | node's searched win prob (side to move) — snapshot `values` |
| `visit_count` | uint32 | node's visits — snapshot `visit_counts` |
| `is_root` | bool | row 0 of each per-move snapshot |
| `outcome` | float32 | final game result from the row's side to move: 1 / 0.5 / 0 |
| `meta` | str (json) | net checkpoint name, sims, noise eps, engine version |

`outcome` per row: parse the FEN's side-to-move field (`fen.split()[1]`);
white win ⇒ 1.0 for `"w"` rows, 0.0 for `"b"` rows; draw ⇒ 0.5.

Put `save_game_shard(path, rows…)` and `load_shard(path)` in `dataset.py`
so selfplay and training share one format definition.

## Step 2 — `selfplay.py`

CLI: `--net PATH` (required; the current-best checkpoint), `--out DIR`,
`--games 100`, `--sims 800`, `--jobs 1`, `--temp-plies 30`,
`--noise-eps 0.25`, `--dirichlet-alpha 0.3`, `--snapshot-min-visits 8`,
`--snapshot-max-depth 30`, `--max-plies 512`, `--seed 0`,
`--workers 2 --batch-size 64` (engine structure).

Per game (seeded `game_seed = seed + game_index`, also used for
`EngineConfig.seed` and the temperature RNG):

```
game = Game(); engine = Engine(EngineConfig(evaluator=TorchEvaluator(net),
                                            workers=…, batch_size=…, seed=game_seed))
engine.set_position(game.fen())
while game.outcome() is None and ply < max_plies:
    result = engine.search(SearchLimits(
        max_time_ms=0, max_simulations=sims, convergence_window=0,
        root_noise_eps=noise_eps, root_dirichlet_alpha=alpha))
    snap = engine.tree_snapshot(min_visits=snapshot_min_visits,
                                max_depth=snapshot_max_depth)
    record rows of snap (fens/values/visit_counts; policy from
        snap.moves[i]/snap.child_visits[i] normalized; is_root = (i == 0))
    move = pick_move(snap.moves[0], snap.child_visits[0], ply, rng)
    game.push(chess.Move.from_uci(move)); engine.advance(move)
engine.close()
outcome per game: game.outcome() (None after ply cap => draw)
fill per-row `outcome`, save shard
```

Details:

- `pick_move`: ply < `temp_plies` ⇒ sample with `p ∝ visits` (τ = 1);
  otherwise argmax. Snapshot row 0 is always the search root (its DFS
  starts there), and `search()` guarantees ≥ 1 root-child visit here.
- Convergence stop is disabled (`convergence_window=0`) — uniform target
  quality (§7.1).
- Tree reuse via `engine.advance` is deliberate; noise is re-applied per
  search (M6a §step 5).
- **Terminal-position rows:** if `snap.moves[i]` is empty, skip the row
  for policy purposes — simplest rule: drop rows with an empty move list
  entirely (they carry no policy target; value knowledge about terminal
  positions is already exact in search).
- `--jobs N`: `multiprocessing.get_context("spawn")`, pool over game
  indices; each worker process builds its own `TorchEvaluator`
  (constructor already sets `torch.set_num_threads(1)`). Games per second
  is the number to print at the end.

## Step 3 — `dataset.py` (training side)

```python
def load_window(data_dir, window: int) -> list[Shard]   # newest `window` games
def sample_batch(shards, batch_size, rng,
                 lambda_root=1.0, lambda_interior=0.0,
                 min_visits_interior=32) -> Batch
```

- Row filter: keep roots always; keep interior rows with
  `visit_count >= min_visits_interior`.
- Value target per row:
  `lam = lambda_root if is_root else lambda_interior`;
  `target = lam * outcome + (1 - lam) * search_value`.
- `Batch` tensors: `planes` float32 `[B,19,8,8]` — built row-by-row with
  `_mcts.encode_planes(fen)` (this is the train/search-identity guarantee,
  §3.4 of the design; do not write a Python encoder) — plus the sparse
  policy target (`policy_index` int64 `[B, Kmax]` padded, `policy_prob`
  float32 `[B, Kmax]` padded with 0) and `value_target` float32 `[B]`.
- Sampling: uniform over the filtered rows of the window. Precompute a
  flat row index once per `load_window`.

## Step 4 — `train.py`

CLI: `--data DIR`, `--in CKPT` (checkpoint to continue from; omit =
fresh random net), `--out CKPT`, `--window 5000`, `--steps 4000`,
`--batch 256`, `--lr 1e-3`, `--weight-decay 1e-4`, `--lambda-root 1.0`,
`--lambda-interior 0.0`, `--min-visits-interior 32`, `--seed 0`,
`--log-every 100`.

- Optimizer: `Adam(lr, weight_decay=weight_decay)`. Fixed LR (schedule is
  an open point in the design — do not add one).
- Loss per batch (model in `train()` mode):
  - value: `F.binary_cross_entropy(value_pred, value_target)`
  - policy: `-(policy_prob * log_softmax(logits).gather(1, policy_index)).sum(1).mean()`
    — padded entries have `policy_prob == 0`, so they contribute nothing
    (make sure padding indices are any valid index, e.g. 0, never −1).
- Log `step, value_loss, policy_loss` every `--log-every`; save the
  checkpoint via `TorchEvaluator.save`-compatible format (arch params +
  state_dict) at the end. Print final average losses — M6c's gate wants
  to see both decrease between generation 0 and 1 data.

## Step 5 — `arena.py`

CLI: `--net-a PATH`, `--net-b PATH`, `--games 100`, `--sims 400`,
`--workers 2 --batch-size 64`, `--temp-plies 4`, `--temp 0.5`,
`--max-plies 512`, `--seed 0`, `--gate 0.55`.

- Game g: A plays white iff `g % 2 == 0`. Both engines: noise **off**
  (`root_noise_eps=0`), fixed sims, convergence off.
- Move choice: first `temp_plies` plies sample `p ∝ visits^(1/temp)` (from
  `tree_snapshot(min_visits=1, max_depth=1)` row 0 — opening variety),
  then argmax.
- Both engines `advance()` every move (each maintains its own tree); ply
  cap adjudicates a draw.
- Score for A: win 1, draw 0.5. Output: `A score: 61.5/100 (0.615) — PROMOTE`
  (or `KEEP`), exit code 0 either way; also print W/D/L split. Gate:
  score ≥ `--gate`.

A **generation** is then (document this in the module docstring of
`training/__init__.py`, no extra driver script):

```
chessengine-selfplay --net best.pt --out data/gen3 --games 500 --jobs 8
chessengine-train --data data --in best.pt --out candidate.pt
chessengine-arena --net-a candidate.pt --net-b best.pt
# promoted? -> cp candidate.pt best.pt
```

Generation 0: create the initial random `best.pt` with a two-liner
(`TorchEvaluator().save("best.pt")`) — give it a `--init` flag on
`train.py` or mention the snippet in the docstring.

## Step 6 — Tests (`tests/python/test_training.py`)

All torch-dependent tests start with `pytest.importorskip("torch")`. Keep
every engine tiny: `blocks=1, filters=8`, `sims=32`, `batch_size=8`,
`workers=1`.

1. **Shard round-trip** (no torch): build synthetic rows, save, load,
   compare — including ragged policy alignment via `row_offsets`.
2. **Outcome perspective** (no torch): a fake game record where white
   won → `"w"` rows get 1.0, `"b"` rows 0.0.
3. **Value-target blend** (no torch): `lambda_root=1, lambda_interior=0`
   ⇒ root rows get `outcome`, interior rows get `search_value`; a 0.5/0.5
   blend interpolates.
4. **Self-play smoke:** 1 game, tiny net, `--max-plies 20` → a shard file
   exists, loads, `is_root.sum() == number of moves recorded`, all
   `policy_prob` rows sum to ≈ 1, all indices in `[0, 4672)`.
5. **Training smoke:** generate 2 tiny games, run `train` for 20 steps,
   batch 16 → finishes, checkpoint saves and reloads, losses are finite.
6. **Arena smoke:** 2 games, tiny nets both sides → completes, score in
   `[0, 2]`, alternated colors (assert via which engine moved first).
7. **End-to-end mini-generation** (`@pytest.mark.slow`): selfplay 2 games
   → train 20 steps → arena 2 games, all through the `main()` entry
   points with `--` args (use `tmp_path`). This is the plumbing gate from
   DESIGN-M6.md §8.

For the real (non-CI) acceptance run, document in the module docstring:
gen-1 net must beat the random-init net at ≥ 55 % over 100 arena games,
and both loss components must decrease during training.

## Definition of done

- [ ] Full suite green (torch-less default install skips torch tests).
- [ ] `chessengine-selfplay/-train/-arena` run end-to-end on a tiny config
      locally (the slow test proves it in CI).
- [ ] Shards from a real self-play run load and re-filter with a different
      `--min-visits-interior` without regeneration (that's why
      `visit_count` is stored).
- [ ] No torch import from `import chessengine`.

## Pitfalls

- Always `engine.close()` (or use the context manager) before dropping an
  engine — the GIL shutdown rule from M6b applies to every script here.
- `tree_snapshot` cannot be called while a search runs — only after
  `search()` returns (the drivers here are all blocking, so this is
  natural; don't switch to `start()`/`stop()` polling).
- Normalize policy targets from `child_visits` per row — snapshot visit
  lists are raw counts and exclude zero-visit children by design.
- In `sample_batch`, padded policy entries need prob 0 **and** a valid
  index; `gather` with −1 crashes.
- Multiprocessing: pass checkpoint *paths* to workers, never model
  objects; "spawn" context avoids forking a process that has a live C++
  evaluator thread.
- Keep λ defaults exactly `root=1.0, interior=0.0` (user decision, design
  §7.3); they are flags for experiments, not knobs to retune silently.

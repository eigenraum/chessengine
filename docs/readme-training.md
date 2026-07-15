# Training the learned evaluator

This is the practical companion to [`docs/design/DESIGN-M6.md`](design/DESIGN-M6.md)
(read that first for *why* things are shaped this way — canonical encoding,
value bootstrapping, the λ blend). This page answers the operational
questions: how to run self-play, how to train, how to continue from a
checkpoint, and how to point the terminal/browser UI at a trained net.

Everything here needs the optional `train` dependency group (torch is not
installed by default):

```sh
uv sync --group train
```

## The pipeline is three separate commands, not a fused loop

Self-play and training are **not** two halves of one loop — they're
independent CLI commands you run one after another, each reading/writing
plain files (`.npz` shards, `.pt` checkpoints) on disk:

```sh
chessengine-selfplay --net best.pt --out data/gen3 --games 500 --jobs 8
chessengine-train --data data --in best.pt --out candidate.pt
chessengine-arena --net-a candidate.pt --net-b best.pt
# candidate scored >= the gate ("PROMOTE")? adopt it as the new best:
cp candidate.pt best.pt
```

That's one **generation**. The three commands stay independent — self-play,
training, and arena have very different resource profiles (CPU search vs.
GPU/CPU gradient steps vs. more CPU search), so keeping them as separate
steps you can schedule, retry, and inspect independently is more useful
than hiding them behind one command — see the module docstring in
[`python/chessengine/training/__init__.py`](../python/chessengine/training/__init__.py)
for the same summary in code. `chessengine-loop` (below) is the driver that
calls all three back to back, generation after generation; it's a thin
wrapper around exactly this four-line sequence, not a different code path.

## Generation 0: a random-initialized checkpoint

Self-play needs a net to search with, even for the very first generation.
Create one random-initialized checkpoint to bootstrap from:

```sh
chessengine-train --init --out best.pt
```

## Self-play: generating training data

```sh
chessengine-selfplay --net best.pt --out data/gen1 --games 500 --jobs 8
```

Each game is searched move by move with root Dirichlet noise (for
exploration) and written as one `.npz` shard under `--out`; every node in
the exported search tree (not just the moves actually played) becomes a
training row (interior-node training, DESIGN-M6.md §7.3), so one game
yields hundreds of rows. A `tqdm` progress bar tracks games completed (not
plies or rows) and, with `--jobs > 1`, ticks per game as each worker process
finishes one — not once per worker's whole assigned batch.

Flags that matter day to day:

| Flag | Default | What it does |
|---|---|---|
| `--net` | *(required)* | current-best checkpoint to self-play with |
| `--out` | *(required)* | directory to write `.npz` shards into (created if missing) |
| `--games` | 100 | games to play |
| `--sims` | 800 | MCTS simulations per move |
| `--jobs` | 1 | parallel worker **processes** (see below) |
| `--workers` | 2 | search **threads** per engine (within one game) |
| `--batch-size` | 64 | evaluator batch size |
| `--temp-plies` | 30 | plies sampled `∝ visits` before switching to argmax |
| `--noise-eps` / `--dirichlet-alpha` | 0.25 / 0.3 | root exploration noise |
| `--snapshot-min-visits` | 8 | interior rows below this visit count aren't exported |
| `--max-plies` | 512 | ply cap; unresolved games at the cap are scored a draw |
| `--seed` | 0 | base seed; game *i* uses `seed + i` (reproducible per game) |
| `--device` | cpu | net device: `auto` picks cuda, then Apple Silicon (mps), then cpu |
| `--parallel-games` | 1 | games run concurrently against one shared net (see below); mutually exclusive with `--jobs` |

`--jobs` vs. `--workers`: `--jobs N` runs N **processes**, each with its own
loaded net and its own set of games (`multiprocessing`, spawn context — a
forked process can't safely inherit a live C++ evaluator thread); `--workers`
is the tree-parallel search **thread** count inside one engine, same knob as
the terminal/browser UI. Scale `--jobs` with your core count; `--workers`
rarely needs to go far past 2–4 for a small net.

`--jobs` vs. `--device`: they answer different questions and don't compose
well. `--jobs` is the CPU story — N processes, each on its own cores. On a
GPU/MPS device, several *processes* sharing it fight over the same context,
so keep `--jobs` low (2–4 at most) once `--device` isn't `cpu`; see
DESIGN-GPU.md §4.3 for the full rationale and `tools/bench_eval.py` for
measuring your own net/hardware before picking a batch size.

`--parallel-games` (DESIGN-GPU.md §5, the "G2" slice): the GPU-shaped
alternative to `--jobs`. Instead of N processes each with their own
evaluator, `--parallel-games N` runs N games as threads *in one process*,
sharing one `EvalServer` that coalesces all their leaf batches into single,
larger forward passes on `--device`. A single engine rarely produces a
GPU-sized batch on its own early in a search; pooling several games' batches
is what actually fills one. Requires `--jobs 1` (the default) — pick one or
the other, they answer the same "how do I parallelize" question for
different hardware. Reproducibility is relaxed here: which games' batches
land together depends on thread timing, so results aren't bit-identical
across runs the way `--jobs 1 --parallel-games 1` (the reference path) is.

Shards store `visit_count` per row, so you can **re-filter an existing data
directory with a different `--min-visits-interior` at training time without
regenerating anything** — that filter lives in `chessengine-train`, not
`chessengine-selfplay` (see below).

## Training: fitting the net to a data window

```sh
chessengine-train --data data --in best.pt --out candidate.pt
```

Reads the most recent `--window` games under `--data`, samples minibatches
uniformly from the filtered rows, and optimizes value (BCE) + policy
(soft-target cross-entropy) loss with Adam. Both loss components print every
`--log-every` steps and again as a final average — they should both trend
down between a fresh checkpoint and one trained on real data.

| Flag | Default | What it does |
|---|---|---|
| `--data` | *(required unless `--init`)* | self-play shard directory |
| `--in` | *(none = fresh random net)* | checkpoint to continue training from |
| `--out` | *(required)* | checkpoint to write |
| `--window` | 5000 | most recent N games to train on |
| `--steps` | 4000 | optimizer steps |
| `--batch` | 256 | minibatch size |
| `--lr` / `--weight-decay` | 1e-3 / 1e-4 | Adam hyperparameters |
| `--lambda-root` / `--lambda-interior` | 1.0 / 0.0 | value-target blend (DESIGN-M6.md §7.3) — leave these alone unless you're deliberately experimenting |
| `--min-visits-interior` | 32 | interior rows need at least this many visits to be trained on |

### Continuing training from a checkpoint

Yes — that's what `--in` is for. `--in CKPT` loads an existing checkpoint's
architecture *and* weights and keeps optimizing from there; omit it to start
from a fresh random net instead. A normal generation always passes
`--in best.pt` (continue from the current best) and writes a new
`--out candidate.pt`, which then has to clear the arena gate before you
promote it.

### `--init`: the generation-0 bootstrap

`--init --out PATH` skips training entirely and just writes a random net —
that's the two-liner from the "Generation 0" section above, exposed as a
flag instead of a Python snippet. `--data`/`--in` are ignored (and not
required) in this mode.

## Arena: deciding whether a candidate is actually better

```sh
chessengine-arena --net-a candidate.pt --net-b best.pt --games 100
```

Plays A vs. B with colors alternating every game, noise off, a short
temperature-sampled opening (`--temp-plies`) for variety, then argmax.
Prints e.g.:

```
A score: 61.5/100 (0.615) — PROMOTE
W 55  D 13  L 32
```

`PROMOTE` means A's score cleared `--gate` (default 0.55 = 55%); `KEEP` means
it didn't. Exit code is always 0 either way — **the promotion itself is a
manual step you decide on** after reading the verdict:

```sh
cp candidate.pt best.pt
```

There's no automatic promotion; this is intentional (DESIGN-M6.md §7.5) —
a bad net silently overwriting `best.pt` is worse than an extra manual
`cp`.

| Flag | Default | What it does |
|---|---|---|
| `--net-a` / `--net-b` | *(required)* | the two checkpoints |
| `--games` | 100 | total games (A plays white on even-indexed games) |
| `--sims` | 400 | simulations per move, noise off |
| `--gate` | 0.55 | promotion threshold |
| `--device` | cpu | net device for both A and B: `auto` picks cuda, then mps, then cpu |
| `--parallel-games` | 1 | games run concurrently, each pair of engines backed by two shared `EvalServer`s (one per net) |

## `chessengine-loop`: automating the pipeline

```sh
chessengine-loop --best best.pt --data data
```

Runs self-play → train → arena generation after generation, forever (until
you Ctrl-C it) or for a fixed `--generations N`. Each generation is exactly
the four-line pipeline above, driven through `chessengine-selfplay`,
`chessengine-train`, and `chessengine-arena`'s own `run()` entry points —
same code path as the manual commands and the tests, not a reimplementation,
so their tqdm progress bars show up exactly as they do standalone. Logging
(one line per stage: games/s, losses, arena verdict) goes to stdout, and
per-generation scalars go to a TensorBoard run:

```sh
tensorboard --logdir runs
```

logged tags: `selfplay/games`, `selfplay/elapsed_s`, `train/value_loss`,
`train/policy_loss`, `arena/score_fraction`, `arena/wins`, `arena/draws`,
`arena/losses`, `arena/promoted`.

If `--best` doesn't exist yet, the loop creates it (`chessengine-train
--init`) before the first generation — no separate generation-0 step
needed. Every generation's self-play output goes into one flat `--data`
directory (not per-generation `data/genN` subdirectories like the manual
example above) since `chessengine-train`'s `--window` scans its data
directory non-recursively for the most recent games by file mtime; one
flat, ever-growing directory is what makes that window slide correctly
across generations.

**Unlike `chessengine-arena` alone, this loop auto-promotes**: a candidate
that clears `--gate` is copied over `--best` immediately, because an
unattended loop has no other way to feed generation N's result into
generation N+1. Pass `--no-auto-promote` to opt back into the manual-`cp`
behavior — every generation then self-plays against the same `--best` net,
and candidates just accumulate on disk for you to inspect.

A candidate that doesn't clear the gate isn't necessarily thrown away:
every non-promoted candidate's checkpoint is **kept on disk (as
`candidate-gen000N-<timestamp>.pt` next to `--best`) as long as its arena
win rate against `--best` clears `--keep-threshold`** (default 0.5 — better
than a coin flip); otherwise its file is deleted right after the arena
verdict, so clearly-worse candidates don't pile up while near-misses stay
around for inspection or as a base to keep training from. Promoted
candidates are always kept.

`--device` and `--parallel-games` (defaults `auto` and `8`) are threaded
through to self-play and arena alike, matching this repo's own benchmarked
recommendation for Apple Silicon (DESIGN-GPU.md's G2 retrospective); pass
`--device cpu --parallel-games 1 --jobs N` instead to use the CPU-process
story (`--jobs`) rather than GPU batch-coalescing.

| Flag | Default | What it does |
|---|---|---|
| `--best` | `best.pt` | current-best checkpoint; created if missing |
| `--data` | `data` | flat shard directory shared by every generation |
| `--generations` | 0 | generations to run; 0 = run until interrupted |
| `--auto-promote` / `--no-auto-promote` | on | copy a PROMOTE-d candidate over `--best` automatically |
| `--keep-threshold` | 0.5 | a non-promoted candidate's checkpoint is deleted unless its arena win rate clears this |
| `--tensorboard-dir` | `runs` | TensorBoard event file directory |
| `--device` | `auto` | net device for self-play, training, and arena alike |
| `--parallel-games` | 8 | games run concurrently against one shared net (self-play + arena) |
| `--jobs` | 1 | self-play worker **processes** (mutually exclusive with `--parallel-games`) |
| `--workers` | 2 | search threads per engine |
| `--batch-size` | 64 | evaluator batch size |
| `--selfplay-games` / `--selfplay-sims` | 100 / 800 | self-play games and sims/move per generation |
| `--max-plies` | 512 | ply cap for both self-play and arena |
| `--train-steps` / `--train-batch` | 4000 / 256 | optimizer steps and minibatch size per generation |
| `--window` | 5000 | most recent N games trained on |
| `--min-visits-interior` | 32 | interior-row visit-count filter |
| `--arena-games` / `--arena-sims` | 100 / 400 | candidate-vs-best arena games and sims/move |
| `--gate` | 0.55 | promotion threshold |
| `--seed` | 0 | base seed; generation *g* offsets self-play/arena seeds by `g * --selfplay-games` |

Every flag not listed here (root noise, temperature plies, snapshot
filters, Adam hyperparameters, …) stays at the underlying command's own
default — `chessengine-loop` only overrides what it explicitly exposes.

## Playing against a trained checkpoint

Both frontends default to the built-in material evaluator; pass
`--evaluator torch --net PATH` to play against a checkpoint instead
(`--net` omitted = a random-weight net, mostly useful for smoke-testing that
the plumbing works, not for actually playing). Add `--device auto` to run
the net on cuda/mps if available — for a single interactive search there's
no other game to batch with, so the win is smaller than in self-play/arena
(DESIGN-GPU.md §8), but it's free.

Terminal:

```sh
chessengine --evaluator torch --net best.pt
```

Browser:

```sh
chessengine-web --evaluator torch --net best.pt
```

Both auto-raise `--workers`/`--batch-size` to 2/64 in torch mode (a net
wants bigger batches to amortize the Python round-trip); pass them
explicitly to override. The web UI's live config panel shows which
evaluator is active (`GET /api/config` → `structural.evaluator`) but
**switching evaluators is a startup-time flag, not something you can change
from the running UI** — restart `chessengine-web` with a different `--net`
to play a different checkpoint.

## Browsing self-play games in the web UI

`chessengine-web`'s **Self-play** tab loads and steps through a `.npz`
shard written by `chessengine-selfplay` — useful for sanity-checking that
self-play is producing sensible games and reasonable policies without
writing a script. Enter the shard's path (server-side; the shard has to be
readable by the machine running `chessengine-web`) and click **Load**:

- The move list on the right replays the game move by move (SAN), click any
  move to jump the board to that position.
- Below it: that position's recorded search value, visit count, and game
  outcome (all from the side to move's perspective), plus the top 8 moves
  by recorded visit probability — the actual policy target that position
  contributes to training.

This is read-only and entirely separate from the live game/engine session —
loading a shard never touches the board you're playing on in the other
tabs, and there's no server-side "current shard" state (each browser tab
that loads a shard keeps its own view of it).

A shard only stores positions and sparse policy targets, not the played
move at each one — the viewer reconstructs it from consecutive root
positions, and decodes the policy's move indices back into UCI moves. Both
are pure computation over data already in the shard; no separate format or
regeneration needed.

## Sanity-checking the whole pipeline locally

Before spending real compute, a few seconds on a tiny net proves the
plumbing works end to end:

```sh
chessengine-train --init --out /tmp/best.pt
chessengine-selfplay --net /tmp/best.pt --out /tmp/data --games 2 --sims 24 \
    --workers 1 --batch-size 8 --max-plies 16 --snapshot-min-visits 1
chessengine-train --data /tmp/data --in /tmp/best.pt --out /tmp/candidate.pt \
    --steps 20 --batch 16 --min-visits-interior 1
chessengine-arena --net-a /tmp/candidate.pt --net-b /tmp/best.pt \
    --games 2 --sims 24 --workers 1 --batch-size 8 --max-plies 16 --temp-plies 2
```

This is exactly what `tests/python/test_training.py`'s smoke tests and its
`@pytest.mark.slow` end-to-end test do — `uv run pytest -m slow` runs the
latter, `uv run pytest` runs the rest.

## Real acceptance criteria

Per `chessengine/training/__init__.py`'s docstring: a real (non-toy) run
should show both loss components decreasing between generation 0 and
generation 1 data, and generation 1's net should beat the random-init net at
≥ 55% over 100 arena games. Nothing in this repo runs that for you
automatically — it's the actual research/compute step the rest of this
pipeline exists to support.

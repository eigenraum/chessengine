# Design Document: M6 — Learned Evaluation

Status: **draft (2026-07-15)**. Companion to `DESIGN.md` (which this refines,
not replaces): M6 is the "learned evaluation" milestone — PyTorch policy/value
evaluator, policy priors in the search, self-play data generation, and the
first training loop.

Guiding rule as ever: educational project — readability and simplicity win
over the last percent of runtime performance.

---

## 1. Decisions Taken

Settled up front so the rest of the document can be concrete:

1. **PyTorch stays in Python.** No LibTorch in the C++ build. The evaluator
   thread crosses into Python once per batch (the single deliberate
   GIL-touching point from DESIGN.md §5).
2. **Dense AlphaZero-style policy output.** The net emits logits over a fixed
   move space of 73 × 64 = 4672 entries (§3.3). C++ owns the move↔index
   mapping and the legal-move masking/softmax; Python only ever sees
   fixed-shape tensors.
3. **Position encoding in C++, canonical side-to-move perspective** (§3.1,
   §3.2). The identical encoder is exposed to Python so training uses the
   exact same input representation — no duplicated encoding logic.
4. **Training on interior tree nodes from the start**, not just root
   positions: interior rows use their own search value as the value target
   (bootstrapping), root rows use the game outcome (§7.3).
5. **Three shippable slices** (§8): M6a policy plumbing (pure C++), M6b
   encoding + Python bridge + net, M6c self-play + training loop. Each leaves
   the engine fully working.

---

## 2. Gap Analysis

What M6 needs that the codebase does not have yet:

| Gap | Where |
|---|---|
| `Evaluator` returns values only — no policy output | `cpp/eval/evaluator.h` |
| Priors hardcoded uniform at expansion | `cpp/mcts/search.cpp` (`maybe_expand`) |
| `Node::prior` is a plain `float` — data race once priors are written after publication | `cpp/mcts/node.h` |
| Evaluator hardwired to `MaterialEvaluator` | `cpp/bindings.cpp` |
| No root Dirichlet noise / no temperature play | `SearchLimits` + self-play driver |
| No position/move encoding, no net, no training code | new: `cpp/eval/encode.*`, `python/chessengine/eval/`, `python/chessengine/training/` |

`SearchConfig::seed` is already reserved for root noise; `tree_snapshot()`
already exports (fen, value, child moves, child visits) per node — exactly the
training rows we need.

---

## 3. Encoding (`cpp/eval/encode.h/.cpp`)

One C++ module owns both encodings (planes and move indices). It is used by
the evaluator bridge at search time and exposed through pybind for the
training pipeline and for tests — a single source of truth.

### 3.1 Canonical perspective

Every position is encoded **from the side to move's point of view**: if black
is to move, ranks are mirrored (a1↔a8; files unchanged) and piece colors are
swapped. The net therefore always answers "win probability for the player to
move" — the same convention every value in the engine already uses — and
never needs to learn color symmetry. Move indices use the same mirroring
(§3.3).

### 3.2 Input planes — `float32 [19, 8, 8]`

Plane `[r, f]` layout with rank/file after canonicalization:

| Planes | Content |
|---|---|
| 0–5 | side-to-move pieces: P, N, B, R, Q, K (one-hot) |
| 6–11 | opponent pieces: P, N, B, R, Q, K |
| 12–13 | side-to-move castling rights: kingside, queenside (constant 0/1) |
| 14–15 | opponent castling rights: kingside, queenside |
| 16 | en-passant target square (one-hot, usually empty) |
| 17 | halfmove clock / 100 (constant plane) |
| 18 | all ones (lets 3×3 convs detect the board edge) |

Deliberately **no history or repetition planes**: nodes carry no game history
(DESIGN.md §4.1), and the search already handles repetition draws by rule.
This costs some strength (the net cannot see looming repetitions) and is an
accepted simplification — revisit only with evidence.

API:

```cpp
namespace eval {
constexpr int PLANES = 19;
constexpr int POLICY_SIZE = 73 * 64;  // 4672

// Writes PLANES*64 floats for `board` into `out` (caller-sized).
void encode_planes(const core::Board& board, std::span<float> out);

// Index of `move` in the policy output for `board`'s side to move.
int move_index(const core::Board& board, core::Move move);
}
```

### 3.3 Move index — 73 move-type planes × 64 from-squares

`index = move_type * 64 + from_square` (from-square canonicalized per §3.1),
matching a policy head that outputs `[73, 8, 8]` and flattens to 4672:

| move_type | Meaning |
|---|---|
| 0–55 | "queen moves": direction (N, NE, E, SE, S, SW, W, NW) × distance 1–7; `type = dir*7 + (dist-1)` |
| 56–63 | knight moves (8 fixed offsets) |
| 64–72 | underpromotions: (capture toward file−1, push, capture toward file+1) × (N, B, R) |

Conventions (AlphaZero standard): queen promotions are encoded as ordinary
queen moves; castling is the king's two-square queen move; en passant is an
ordinary diagonal pawn move. Every legal move in every reachable position
maps to exactly one index — a property the tests check over the perft corpus.

### 3.4 Python exposure

```python
_mcts.encode_planes(fen: str) -> np.ndarray            # float32 [19, 8, 8]
_mcts.move_indices(fen: str, ucis: list[str]) -> list[int]
```

Used by `training/dataset.py` (encode positions and sparse policy targets at
training time) and by the encoding tests, which cross-check against an
independent pure-Python reference built on python-chess — same philosophy as
the perft gate.

---

## 4. Evaluator Interface v2 — policy through the queue

### 4.1 The interface

`evaluate` moves from parallel spans to a batch of small request structs, each
carrying an optional policy slot:

```cpp
namespace eval {

// One position in an evaluation batch. `moves`/`priors_out` have equal
// length and are non-empty only when the caller wants a policy for this
// position; priors_out[i] receives P(moves[i] | position), summing to 1.
// Values remain win probabilities in [0, 1] for the side to move.
struct EvalRequest {
    const core::Board* board;
    std::span<const core::Move> moves;
    std::span<float> priors_out;
    float* value_out;
};

class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void evaluate(std::span<const EvalRequest> batch) = 0;
};

}  // namespace eval
```

- **Masking/softmax lives in the evaluator implementation.** The search hands
  in legal moves and gets back a normalized prior distribution; how it is
  produced (uniform, gathered NN logits + softmax) is the evaluator's
  business.
- `MaterialEvaluator` writes uniform priors — behavior is bit-identical to
  today, which is the M6a regression gate.
- `EvalQueue` changes mechanically: `Request` grows the two spans, the
  evaluator thread assembles `EvalRequest` batches. Buffers backing the spans
  live in the worker's stack frame, which is parked inside
  `EvalQueue::evaluate()` for the duration — no lifetime subtleties.

### 4.2 Search-side changes: priors are written back after evaluation

Expansion happens *before* evaluation (the leaf's children are published with
placeholder priors, the path parks on its virtual loss). The policy therefore
arrives late and is **written back** into the already-visible children:

1. In `descend`, when a simulation parks at an evaluation leaf whose node it
   expanded (or found `EXPANDED`), the worker copies the children's moves
   into that in-flight simulation's move buffer and points the request's
   `moves`/`priors_out` at it. If the node is not `EXPANDED` (lost race,
   arena full), the request is value-only — exactly today's semantics.
2. After `queue_.evaluate(...)` returns, the worker stores `priors_out[i]`
   into child `i`'s `prior` field, then backpropagates the value as before.

Consequences, all acceptable:

- **Placeholder priors are visible briefly.** Between expansion and prior
  write-back, concurrent simulations select among children with uniform
  priors. Self-correcting within a few visits; this is exactly what engines
  with delayed expansion live with permanently.
- **Double writes.** Two workers can park the same leaf (expansion-race
  loser) and both write priors — identical values, harmless.
- **`Node::prior` becomes `std::atomic<float>`** (relaxed loads/stores):
  it is now written concurrently with `select_child` reads. Same 4 bytes,
  node stays 32 bytes, and the TSan gate stays honest.
  `maybe_expand` initializes it to uniform as the placeholder.

### 4.3 Root handling: synchronous evaluation + Dirichlet noise

`run_controller` already expands the root before launching workers. M6a
extends that sequence:

1. `maybe_expand(root)` — as today.
2. **Evaluate the root synchronously** through the queue (batch of one) and
   write its children's priors. One evaluator call per search — negligible,
   and it removes the "root explored on uniform priors until the first batch
   returns" wrinkle. The root value is discarded (priors only); simulation
   counts are unaffected.
3. **Apply Dirichlet noise** if enabled:
   `p_i ← (1−ε)·p_i + ε·d_i`, `d ~ Dir(α)` sampled with an RNG seeded from
   `SearchConfig::seed` and a per-search counter — `workers=1` stays fully
   deterministic, noise included.

New `SearchLimits` fields (per-search, because self-play wants noise and
interactive play does not):

```cpp
float root_noise_eps = 0.0f;        // 0 = off (play/analysis)
float root_dirichlet_alpha = 0.3f;  // chess-standard concentration
```

Note on tree reuse: `advance()` keeps the subtree, so a reused root already
has NN priors; step 2 simply refreshes them and step 3 re-applies fresh noise
per search — matching AlphaZero, which draws new root noise every move.

---

## 5. Python Evaluator Bridge (`bindings.cpp`)

A `PyEvaluator : eval::Evaluator` living in the bindings — the one place
where C++ calls Python:

```
callback: (planes: np.float32 [N, 19, 8, 8])
       -> (values: np.float32 [N], policy_logits: np.float32 [N, 4672])
```

Per batch, on the evaluator thread:

1. Encode all boards into a reusable float buffer (no GIL needed).
2. **One `gil_scoped_acquire` scope:** wrap the buffer as a numpy array, call
   the callback, copy `values` out, and gather each request's legal-move
   logits (via `move_index`) into per-request scratch vectors.
3. Outside the GIL: softmax each gathered vector into `priors_out`.

Rules and properties:

- Workers never touch Python; only the evaluator thread acquires the GIL,
  once per batch (DESIGN.md §5 upheld).
- **No deadlock by construction:** every binding that can block on the search
  (`search`, `stop`, `tree_view`) already releases the GIL; `stats()` never
  blocks. New rule for the boundary: any binding that can wait on search
  progress must release the GIL — with a Python evaluator this is now
  load-bearing, not just polite.
- Values keep the engine-wide convention: win probability in [0, 1] for the
  side to move (which is also the canonical encoding perspective — no flips
  anywhere in the bridge).

Construction: `_mcts.Engine(config)` keeps the material evaluator;
`_mcts.Engine(config, callback)` uses the bridge. On the Python side,
`EngineConfig` gains `evaluator: Callable | None = None`.

**Batching config for NN mode:** batches form from in-flight simulations
(each worker parks up to `batch_size`), so even `workers=1` produces full
`batch_size` batches — the sequential reference mode works unchanged with a
net. Sensible NN defaults to start from: `batch_size=64`, `workers=2–4`
(CPU); the material default stays `batch_size=8`.

---

## 6. The Network (`python/chessengine/eval/torch_eval.py`)

Small AlphaZero-style ResNet — CPU inference is the bottleneck, so start
tiny and let arena results justify growth:

- **Trunk:** 3×3 conv 19→64 + BN + ReLU, then 4 residual blocks
  (64 filters, two 3×3 convs + BN each, skip connection).
- **Policy head:** 1×1 conv 64→73 → flatten to 4672 raw logits
  (index = plane·64 + square, matching §3.3). No softmax in the net —
  masking + softmax happen in C++ over legal moves only.
- **Value head:** 1×1 conv 64→8 + BN + ReLU → flatten → FC 512→128 + ReLU →
  FC 128→1 → sigmoid. Output is the win probability in [0, 1].

`TorchEvaluator` wraps the model as the callback: `eval()` mode,
`torch.no_grad()`, `torch.from_numpy` in, `.numpy()` out, single-threaded
torch by default (`torch.set_num_threads(1)`) so self-play workers can be
parallelized at the process level instead. Constructor takes a checkpoint
path or `None` for random initialization.

Model size/blocks/filters are constructor parameters — the checkpoint stores
them so `TorchEvaluator` can rebuild the right architecture from the file
alone.

---

## 7. Self-Play & Training (`python/chessengine/training/`)

```
training/
├── selfplay.py   # generate games -> .npz shards
├── dataset.py    # shards -> shuffled training tensors (uses _mcts encoders)
├── train.py      # optimize the net on a window of recent games
└── arena.py      # net-vs-net matches, promotion gate
```

Driven by console scripts (`chessengine-selfplay`, `chessengine-train`,
`chessengine-arena`); a generation is: selfplay → train → arena → promote.

### 7.1 Self-play driver

Per game: `Game` (python-chess, source of truth) + one `Engine` with the
current-best net. Per move:

1. `search(limits)` with fixed `max_simulations` (default 800; convergence
   stop disabled for reproducible target quality), `root_noise_eps = 0.25`.
2. Record `tree_snapshot(min_visits=snapshot_min_visits, max_depth=…)`.
3. Move selection by **temperature** from the root's child visit counts
   (snapshot row 0): plies 1–30 sample `∝ visits^(1/τ)` with τ = 1, after
   that argmax. Temperature lives entirely in Python — no C++ change.
4. `game.push(move)`; `engine.advance(move)` (tree reuse across self-play
   moves is fine — noise is re-applied per search).

Game ends by python-chess rules; a ply cap (default 512) adjudicates as a
draw. No resignation initially (open point).

### 7.2 Data format — one `.npz` per game

Positions are stored as **FENs, not planes** (shards stay small; planes are
recomputed at training time through `_mcts.encode_planes`, guaranteeing
train/search encoding identity). Policy targets are sparse:

| Field | Content |
|---|---|
| `fens` | one per exported node (all snapshots of the game concatenated) |
| `policy_index`, `policy_prob`, `row_offsets` | ragged sparse policy target per row: normalized child visit counts at `move_index` positions |
| `search_value` | float32, the node's searched win probability (side to move) |
| `visit_count` | uint32, the node's visit count (enables dataset-side re-filtering) |
| `is_root` | bool, row was a search root (a position actually played through) |
| `outcome` | float32, final game result from the row's side-to-move perspective: 1 / 0.5 / 0 |
| meta | net generation, sims per move, engine/config versions |

### 7.3 Training targets

- **Policy target:** normalized child visits — for every exported node,
  root and interior alike.
- **Value target:** `λ·outcome + (1−λ)·search_value` with **per-kind λ**:
  - root rows: `λ_root = 1.0` — the game actually passed through these
    positions; outcome is the classic AlphaZero `z`.
  - interior rows: `λ_interior = 0.0` — the game never visited most of these
    positions, so the outcome is weakly attributable; their own search value
    is the honest target (value bootstrapping).
  Both λs are config, because this blend is exactly the kind of knob worth
  experimenting with in an educational project.
- **Filtering:** interior rows require `visit_count ≥ min_visits` (default
  32 — low-visit values are noise); root rows are always kept. Set by the
  `snapshot_min_visits` passed at generation time, refined by a dataset-side
  filter so old shards can be re-filtered without regeneration.

This is the payoff of decision 4: one self-play game yields hundreds of
training rows instead of ~80, at the cost of noisier interior targets — the
λ split and the visit filter are the two dials that manage that noise.

### 7.4 Training loop (`train.py`)

Plain and standard: sample uniformly from a sliding window of the most recent
N games (default 5000), loss =
`BCE(value_pred, value_target) + CrossEntropy(policy_logits, sparse_policy) + weight_decay`,
Adam, fixed LR (schedule is an open point), batch 256, a few thousand steps
per generation. Reports both loss components separately.

### 7.5 Arena & promotion (`arena.py`)

Candidate vs current-best: fixed simulations per move (noise off), τ = 0.5
for the first 4 plies (opening variety), alternating colors, default 100
games. Candidate promotes at **≥ 55 % score**. Generation 0's "current best"
is the random-initialized net; beating it is M6c's acceptance test. (Beating
the material evaluator is a milestone to *track* in the arena, not a gate —
no promise on when it falls.)

Throughput note: self-play parallelizes over games with `--jobs N`
(multiprocessing; each process owns an engine + net, torch single-threaded).
Within a game, tree parallelism + batching already apply.

---

## 8. Milestones (shippable slices)

### M6a — policy plumbing (pure C++, no torch)

Evaluator interface v2 (§4.1), queue rework, prior write-back with
`atomic<float> prior` (§4.2), synchronous root eval + Dirichlet noise (§4.3),
material evaluator on the new interface, new `SearchLimits` fields through
the bindings and `engine.py`.

*Gates:* with noise off, fixed-seed sequential search is **bit-identical to
pre-M6a** (uniform priors, same order — the whole refactor is behaviorally
invisible); noise on: root priors change, still sum to 1, `workers=1` with
fixed seed stays deterministic; existing parallel statistical tests pass;
ThreadSanitizer clean.

### M6b — encoding, bridge, net

`encode.h/.cpp` (§3) + pybind exports, `PyEvaluator` bridge (§5),
`torch_eval.py` net (§6), CLI flag to pick the evaluator
(`--evaluator material|torch --net PATH`).

*Gates:* plane encoding matches an independent python-chess reference over
the perft corpus; move indices are unique and in-range for all legal moves in
the corpus, and mirror-consistent between color-flipped positions; a fake
Python callback with uniform logits reproduces uniform priors exactly, a
crafted-logits callback yields the expected softmax; a full CLI game plays
with a random-weight net; polling `stats()` from Python while an NN search
runs neither blocks nor deadlocks.

### M6c — self-play + training

`training/` package (§7), npz shard format, generation driver, first trained
net.

*Gates:* end-to-end mini-loop in pytest (tiny net, few games, few sims —
plumbing smoke test); on a real run: both loss components decrease, and
generation 1 beats the random-init net in the arena at ≥ 55 %.

---

## 9. Open Points (deliberately deferred)

- Learning-rate schedule, replay-window size, and the λ blend (§7.3) —
  tuning questions; start with the stated defaults and measure.
- Resignation in self-play (saves time, risks value blind spots) — off until
  game generation is demonstrably throughput-bound.
- GPU inference — the batch interface already fits; nothing in M6 may
  preclude it (CLAUDE.md), but CPU is the target for the first generations.
- History/repetition planes — accepted omission (§3.2); revisit with
  arena evidence.
- Net growth (blocks/filters) — driven by arena results once the loop runs.

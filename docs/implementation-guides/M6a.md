# Implementation Guide: M6a — Policy Plumbing (pure C++)

Implements `docs/design/DESIGN-M6.md` §4 (read it first, plus §1–§2 for
context). Pure C++ + bindings + `engine.py` — **no torch, no encoding** in
this slice. When you are done, the engine behaves exactly as before with the
new options off; the new options are root Dirichlet noise and an evaluator
interface that can carry a policy.

Ground rules:

- Work in this worktree; do not touch files outside the listed set without
  good reason.
- Build after C++ changes: `uv sync --reinstall-package chessengine`
- Test: `uv run pytest`
- After all steps: run the ThreadSanitizer gate (README.md § Development).
- Match the existing code style: comments explain constraints, not mechanics.

Files you will touch:

```
cpp/eval/evaluator.h        # interface v2
cpp/eval/material.h/.cpp    # port to interface v2
cpp/mcts/eval_queue.h       # carry EvalRequest instead of (board, value)
cpp/mcts/node.h             # prior -> atomic<float>
cpp/mcts/tree.cpp           # prior copy must use load/store
cpp/mcts/search.h/.cpp      # move buffers, prior write-back, root eval + noise
cpp/bindings.cpp            # new SearchLimits fields
python/chessengine/engine.py# new SearchLimits fields
tests/python/test_policy.py # new
```

---

## Step 0 — Pin current behavior BEFORE changing anything

Create `tests/python/test_policy.py` with a regression pin: the whole M6a
refactor must be behaviorally invisible while noise is off (priors stay
uniform, produced in the same order), so a fixed-seed sequential search must
return identical results before and after.

```python
"""M6a policy plumbing tests (DESIGN-M6.md section 4)."""

import pytest

from chessengine.engine import Engine, EngineConfig, SearchLimits

PIN_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "8/2k5/8/8/8/3QK3/8/8 w - - 0 1",
]


def _fixed_search(fen: str):
    engine = Engine(EngineConfig(workers=1, seed=42))
    engine.set_position(fen)
    result = engine.search(
        SearchLimits(max_time_ms=0, max_simulations=2000, convergence_window=0)
    )
    return result.best_move, result.simulations, result.nodes, result.root_cp


@pytest.mark.parametrize("fen", PIN_FENS)
def test_sequential_search_pinned(fen):
    # Fill EXPECTED by running once on the unmodified code (step 0), then
    # never change these numbers during M6a: with noise off the refactor
    # must be bit-identical.
    assert _fixed_search(fen) == EXPECTED[fen]
```

Run `_fixed_search` once per FEN on the **unmodified** code, paste the
results as the `EXPECTED` dict, and confirm the test passes twice in a row
(it must be deterministic). Commit this before the refactor. Keep it green
at every subsequent step.

---

## Step 1 — `Node::prior` becomes `std::atomic<float>`

In `cpp/mcts/node.h`:

```cpp
std::atomic<float> prior{0.0f};  // P(move | parent); written back after NN
                                 // evaluation, so concurrent with reads
```

Add nearby: `static_assert(std::atomic<float>::is_always_lock_free);` and
`static_assert(sizeof(Node) == 32);` (the second may already exist — keep
it true).

Fix all readers/writers (atomics are not copyable/assignable the plain way):

- `search.cpp` `maybe_expand`: `child.prior.store(prior, std::memory_order_relaxed);`
- `search.cpp` `select_child`: `child.prior.load(std::memory_order_relaxed)`
- `search.cpp` `tree_view`: `node.prior.load(std::memory_order_relaxed)`
- `tree.cpp` subtree copy: `dst.prior.store(src.prior.load(std::memory_order_relaxed), std::memory_order_relaxed);`

Build; full suite green (the pin test especially).

## Step 2 — Evaluator interface v2

Replace the interface in `cpp/eval/evaluator.h` (keep the cp↔probability
helpers unchanged):

```cpp
// One position in an evaluation batch. `moves` and `priors_out` have equal
// length and are non-empty only when the caller wants a policy for this
// position: priors_out[i] receives P(moves[i] | position), summing to 1.
// *value_out receives the win probability in [0, 1] for the side to move.
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
```

Port `MaterialEvaluator` (`material.h/.cpp`): same value computation as
today; when `moves` is non-empty, fill `priors_out` uniformly with
`1.0f / moves.size()`.

## Step 3 — `EvalQueue` carries requests

In `cpp/mcts/eval_queue.h`:

- Public API becomes `void evaluate(std::span<const eval::EvalRequest> requests);`
  (same blocking semantics, one `remaining` counter per caller as today).
- Internal `Request` becomes `{ eval::EvalRequest req; int* remaining; }`.
- `run()` assembles `std::vector<eval::EvalRequest>` from the drained batch
  and calls `evaluator_.evaluate(batch)`; the "copy results back under the
  mutex" step disappears for values (the evaluator writes through the
  pointers/spans directly), but decrementing `remaining` and notifying
  stays exactly as today.

Lifetime argument (add as a comment): the spans and pointers inside an
`EvalRequest` point into the calling worker's buffers, and the worker is
blocked inside `evaluate()` until `remaining == 0` — so they outlive the
evaluator call by construction.

## Step 4 — Search: collect leaf moves, write priors back

In `search.h/.cpp`:

1. `descend` gains an out-parameter `std::vector<std::vector<core::Move>>& out_moves`
   parallel to `paths`/`out_boards`. At the "fresh leaf" park point
   (`search.cpp`, the `arrived_at_leaf || !expanded` branch): if the leaf
   node's `expand_state` is `EXPANDED` (acquire load), copy its children's
   moves — `tree[node.first_child + i].move` for `i = 0..num_children-1` —
   into a moves vector; otherwise leave it empty (value-only request:
   expansion-race loser or arena full, same semantics as today).
2. `worker_loop` keeps parallel per-batch vectors: `paths`, `boards`,
   `moves_bufs`, `priors_bufs` (`std::vector<std::vector<float>>`), sized
   per descent (`priors_bufs[i].assign(moves_bufs[i].size(), 0.0f)`).
   **Build the `std::vector<eval::EvalRequest>` only after all descents for
   the batch are done** (so no vector reallocation can happen between
   taking a span and using it), then call `queue_.evaluate(requests)`.
3. After the call returns, for each request `i` with non-empty moves:
   `leaf = paths[i].back();` then for each child `j`:
   `tree[leaf.first_child + j].prior.store(priors_bufs[i][j], relaxed);`
   Child order equals move order — children were created from the same
   `generate_legal` list in `maybe_expand`, and your moves buffer read them
   back in index order. Then backprop exactly as today (keep the
   `1.0f - values[i]` perspective flip).

Two workers can both write the same leaf's priors (race loser parked the
same node) — identical values, harmless; say so in a comment.

## Step 5 — Root: synchronous evaluation + Dirichlet noise

New `SearchLimits` fields (in `search.h`, mirroring DESIGN-M6.md §4.3):

```cpp
float root_noise_eps = 0.0f;        // Dirichlet noise weight at the root; 0 = off
float root_dirichlet_alpha = 0.3f;  // concentration; 0.3 is chess-standard
```

In `run_controller`, after `maybe_expand(tree_->root(), ...)` and inside the
`num_children > 0` branch, **before launching workers**:

1. Build one `EvalRequest` for the root board: moves = the root children's
   moves (read as in step 4), priors into a local buffer, value into a
   dummy float. Call `queue_.evaluate(...)` with that single request.
   Store the priors into the root children. The value is discarded — root
   priors only; no visit counts change.
2. If `limits.root_noise_eps > 0`: sample `d ~ Dir(alpha)` (one
   `std::gamma_distribution<double>(alpha, 1.0)` draw per child, normalize
   by the sum; if the sum is not `> 0`, skip the noise) and store
   `p_i = (1 - eps) * p_i + eps * d_i` into the children.

RNG: add a `uint64_t searches_started_ = 0;` member, incremented in
`start()`. Seed per search:
`std::mt19937_64 rng(config_.seed ^ (0x9E3779B97F4A7C15ULL * (searches_started_)));`
This keeps `workers=1` fully deterministic, noise included, and gives fresh
noise each search (which is per-move in self-play, thanks to tree reuse
calling a new search per move).

## Step 6 — Bindings + `engine.py`

- `bindings.cpp`: `def_readwrite` for `root_noise_eps` and
  `root_dirichlet_alpha` on `SearchLimits`.
- `engine.py`: add the two fields to the `SearchLimits` dataclass
  (`root_noise_eps: float = 0.0`, `root_dirichlet_alpha: float = 0.3`) and
  pass them through in `_cxx_limits`.

## Step 7 — Tools

`cpp/tools/search_stress.cpp` constructs `MaterialEvaluator` + `Search`; it
should compile unchanged, but verify by building the TSan target.

## Step 8 — New tests (extend `tests/python/test_policy.py`)

Use `engine.tree_view()` — `prior` per node is already exposed. Root
children are the rows with `parent == 0` (row 0 is the root).

1. **Uniform without noise:** after a short search, root-child priors are
   all equal and sum to ≈ 1 (`pytest.approx`, tolerance 1e-4 — priors are
   rounded to 4 digits in `tree_view`).
2. **Noise changes priors:** same search with
   `root_noise_eps=0.25` → priors are not all equal, still sum to ≈ 1,
   all strictly positive.
3. **Noise is deterministic per seed:** two engines with the same
   `EngineConfig.seed` produce identical root priors under noise; a
   different seed produces different ones.
4. **Sequential determinism survives noise:** two identical
   `workers=1, seed=…, max_simulations=N` searches with noise on return the
   same `best_move` and `pv`.
5. **Parallel smoke:** one noise-on search with `workers=4` completes and
   its root priors sum to ≈ 1 (real race coverage comes from TSan).

## Definition of done

- [ ] Full suite green, **including the untouched step-0 pin test**.
- [ ] New policy tests green.
- [ ] TSan stress run clean (README.md § Development commands), which now
      exercises the prior write-back path.
- [ ] `git grep -n "float prior"` shows no non-atomic prior left.
- [ ] No behavior change with `root_noise_eps = 0` — that is the review
      argument for the whole slice.

## Pitfalls

- `std::atomic<float>` breaks copy/assignment sites — the compiler will
  find them; fix with explicit load/store (tree.cpp is the non-obvious one).
- Do **not** backprop the root evaluation value in step 5 — priors only.
- Build `EvalRequest` spans only after all per-batch buffers stopped
  growing (step 4.2).
- The value-only path (empty `moves`) must keep working — it is exercised
  whenever expansion races or the arena fills.
- Keep `simulations_.fetch_add` and the `1.0f - values[i]` flip in
  `worker_loop` exactly as they are; only the request plumbing changes.

# Design Document: MCTS Chess Engine

Status: **accepted (2026-07-12)**. Companion to `CLAUDE.md`, which holds the
agreed high-level architecture and principles. This document makes the design
concrete: module boundaries, data structures, threading model, and APIs.

Guiding rule (from CLAUDE.md): educational project — readability and simplicity
win over the last percent of runtime performance.

---

## 1. Repository Layout

```
chessengine/
├── pyproject.toml            # uv-managed; scikit-build-core builds the C++ extension
├── CMakeLists.txt
├── CLAUDE.md / DESIGN.md
├── python/chessengine/       # the Python package
│   ├── game.py               # Game: wraps python-chess, source of truth for the game
│   ├── ui/
│   │   └── cli.py            # CLI frontend (board + live search stats)
│   ├── engine.py             # thin Python wrapper around the pybind module
│   └── eval/
│       └── torch_eval.py     # (later) PyTorch policy/value evaluator
├── cpp/
│   ├── core/                 # chess rules, no search knowledge
│   │   ├── board.h/.cpp      # position representation, copy-make apply
│   │   ├── movegen.h/.cpp    # legal move generation, perft
│   │   ├── attacks.h         # leaper tables + classical slider rays
│   │   └── types.h           # Square, Move, Color, ...
│   ├── mcts/
│   │   ├── node.h            # Node layout
│   │   ├── tree.h/.cpp       # arena storage, re-rooting
│   │   ├── search.h/.cpp     # worker loop, PUCT selection, backprop
│   │   └── eval_queue.h/.cpp # batched evaluation queue
│   ├── eval/
│   │   └── material.h/.cpp   # cheap material heuristic
│   └── bindings.cpp          # pybind11 module `_mcts`
└── tests/
    ├── python/               # pytest: game, GUI logic, engine API
    └── cpp/                  # perft, node/tree unit tests (also runnable via pytest)
```

Dependency direction: `ui → game → engine (pybind) → mcts → core`. Nothing
points back up. The GUI imports only `game` and `engine`; swapping it touches
nothing else.

---

## 2. Python Layer

### 2.1 `Game` (game.py)

Thin wrapper around `python-chess`. Source of truth for the real game.

```python
class Game:
    def legal_moves(self) -> list[chess.Move]
    def push(self, move: chess.Move) -> None      # raises on illegal move
    def fen(self) -> str
    def outcome(self) -> chess.Outcome | None
    def turn(self) -> chess.Color
```

No engine or UI knowledge. The engine receives positions from `Game` as FEN
strings — simple, debuggable, and the cost (parsing one FEN per search) is
irrelevant.

### 2.2 Engine wrapper (engine.py)

Pythonic facade over the pybind module; owns the config defaults.

```python
@dataclass
class SearchLimits:
    max_time_ms: int = 5000
    max_simulations: int | None = None
    # convergence: stop early when BOTH hold over the window
    convergence_window: int = 2000       # simulations
    convergence_cp_threshold: int = 5    # max centipawn drift within window

@dataclass
class EngineConfig:
    workers: int = 1                     # 1 = sequential reference mode
    batch_size: int = 8                  # max leaves per evaluation batch
    c_puct: float = 1.5
    virtual_loss: int = 1
    seed: int = 0

class Engine:
    def __init__(self, config: EngineConfig): ...
    def set_position(self, fen: str) -> None          # drops the tree
    def advance(self, uci_move: str) -> None          # re-roots: keeps subtree
    def search(self, limits: SearchLimits) -> SearchResult   # blocking
    def start(self, limits: SearchLimits) -> None     # non-blocking, for GUI
    def stop(self) -> SearchResult                    # interrupt / collect
    def stats(self) -> SearchStats                    # safe to call while running
    def tree_snapshot(self) -> TreeSnapshot           # training data export
```

```python
@dataclass
class SearchStats:            # cheap, lock-free reads of atomics
    simulations: int
    nodes: int
    root_value: float         # win prob from side-to-move's view, 0..1
    root_cp: int              # logistic mapping of root_value to centipawns
    best_move: str            # UCI
    pv: list[str]
    elapsed_ms: int

@dataclass
class SearchResult(SearchStats):
    stop_reason: str          # "time" | "converged" | "interrupted" | "simulations"

@dataclass
class TreeSnapshot:           # one row per exported node (>= min_visits,
    fens: list[str]           # <= max_depth plies below the root)
    visit_counts: np.ndarray  # uint64 per node
    values: np.ndarray        # float32: win prob for the side to move in fens[i]
    moves: list[list[str]]    # explored child moves per node (UCI)
    child_visits: list[list[int]]  # visit distribution over those moves
                                   # (the policy target)
```

### 2.3 CLI GUI (ui/cli.py)

- Renders the board (unicode pieces), move history, and prompt.
- While the engine thinks: calls `engine.stats()` a few times per second and
  redraws a status line — `sims: 84_231  nodes: 61_004  eval: +0.42 (+34cp)  pv: e4 e5 Nf3 …`.
- Input: UCI/SAN moves, plus commands (`quit`, `go`, `stop`, `new`).
- Holds no game state of its own; everything comes from `Game`/`Engine`.

Main loop (human vs engine):

```
render(game) → human move → game.push() → engine.advance(move)
→ engine.start(limits) → poll stats & render until done → engine.stop()
→ game.push(best) → engine.advance(best) → repeat
```

---

## 3. C++ Core (`cpp/core`)

Board representation and move generation, independent of search.

- **Bitboards** (one `uint64_t` per piece type per color) plus a mailbox array
  for piece lookup by square. Standard, well-documented approach.
- **Copy-make**: `Board` is a small value type (~100 bytes); applying a move
  copies the board. Simpler and less error-prone than make/unmake with undo
  stacks, and MCTS descends a path once per simulation, so copy-make costs
  little here. (This is the readability-over-last-percent tradeoff, made
  deliberately.)
- **Move generation**: pre-computed attack tables for knights/kings/pawns;
  classical blocker-loop sliding attacks for bishops/rooks/queens (no magic
  bitboards initially — they're an optimization we can drop in later behind
  the same function signature).
- Draw handling in search: 50-move rule and insufficient material detected in
  C++; threefold repetition approximated by twofold within the search path
  (exact repetition tracking is game-level and lives in python-chess).
- **Validation: perft.** The C++ movegen must reproduce known perft counts and
  match python-chess node-for-node on a corpus of tricky positions (castling,
  en passant, promotions, pins). This test gate is non-negotiable and runs in CI.

---

## 4. MCTS (`cpp/mcts`)

### 4.1 Node layout (node.h)

Nodes live in an **arena** (chunked vector owned by `Tree`); references between
nodes are 32-bit indices, not pointers. Children of a node are **contiguous**
in the arena — selection loops scan a cache-friendly range.

```cpp
struct Node {                          // 32 bytes, one cache line holds 2
    std::atomic<uint32_t> visits;      // N
    std::atomic<int32_t>  virtual_loss;
    std::atomic<double>   value_sum;   // W, from side-to-move's view (fetch_add via CAS)
    float    prior;                    // P (uniform now; NN policy later)
    uint32_t first_child;              // index into arena; 0 = not expanded
    uint16_t num_children;
    uint16_t move;                     // move that led here (from parent)
    std::atomic<uint8_t> expand_state; // UNEXPANDED / EXPANDING / EXPANDED
};
```

Notes:
- Q(child) = W/N is derived, not stored. PUCT: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`.
- `expand_state` is a tiny per-node spinlock for expansion: the winning thread
  (CAS UNEXPANDED→EXPANDING) generates moves, allocates the child range,
  publishes `first_child/num_children`, then sets EXPANDED. Losers treat the
  node as a leaf for this simulation (evaluate it again — harmless).
- Boards are **not stored in nodes**. Each simulation carries a `Board` down
  from the root, applying moves as it descends (copy-make). This keeps nodes
  small and the tree memory-bounded.
- Arena allocation is a single atomic bump pointer; chunks avoid reallocation
  so existing node references stay valid while other threads read them.

### 4.2 Simulation loop (search.cpp)

Each worker thread keeps up to `batch_size` simulations in flight and repeats
until told to stop:

```
1. descend (repeat up to batch_size times):
     from root: child = argmax PUCT (using N + virtual_loss in place of N),
     child.virtual_loss += vl, board.apply(child.move), ...
     - terminal or draw leaf → value from the game result, backprop at once
     - evaluation leaf → expand it, park the path (its virtual loss holds it)
2. submit ALL parked leaves to the EvalQueue in one blocking call
3. backprop each returned value: for n in reversed(path): n.visits++,
   n.value_sum += value (sign-flipped per ply), n.virtual_loss -= vl
```

Parking several simulations per worker is what makes evaluation batches form,
and it amortizes the queue handshake — one condvar round-trip per batch
instead of per simulation (~3.4x sequential throughput with the material
evaluator).

- Virtual loss makes concurrent workers repel each other; it is fully removed
  during backprop, so `workers=1` reproduces textbook sequential MCTS exactly.
- With `seed` fixed and `workers=1`, the search is **deterministic** — this is
  the reference mode used to validate the parallel implementation.

### 4.3 Batched evaluation (eval_queue.h)

One mechanism for all evaluators, per CLAUDE.md:

```cpp
class Evaluator {                       // implemented by material now, NN later
public:
    // positions.size() <= batch_size; writes one value per position (0..1,
    // side-to-move's view) and optionally a policy over legal moves
    virtual void evaluate(std::span<const Board> positions,
                          std::span<float> values_out,
                          PolicyOut* policy_out /*nullable*/) = 0;
};
```

- Workers push their whole in-flight batch into the queue in one blocking
  `evaluate(boards, values)` call. Blocking is fine: virtual loss is already
  applied, and with NN evaluation the evaluator is the bottleneck anyway.
  Simple beats clever here.
- A dedicated **evaluator thread** drains the queue: it takes up to
  `batch_size` requests per pass — no flush timeout needed, whatever
  accumulated while it was busy forms the next batch — calls
  `Evaluator::evaluate`, and wakes the waiting workers.
- **Material heuristic** (`eval/material.h`): piece values + small mobility
  term, squashed to 0..1 via the logistic centipawn mapping. Runs inside the
  evaluator thread; batch is a trivial loop. Uniform priors over legal moves.
- **PyTorch evaluator** (later): the evaluator thread acquires the GIL once
  per batch, hands positions to Python as a tensor, gets values+policy back.
  Workers never touch Python. Same queue, same interface — swapping
  evaluators changes one constructor argument.

### 4.4 Search control & convergence

A lightweight controller (checked by workers every few simulations, and by a
timer) stops the search when the first of these fires:

1. **Time** — `max_time_ms` elapsed.
2. **Simulations** — optional `max_simulations` reached.
3. **Converged** — over the last `convergence_window` simulations, BOTH:
   root centipawn evaluation drifted less than `convergence_cp_threshold`,
   AND the most-visited root child did not change.
4. **Interrupted** — `stop()` called from Python (GUI / Ctrl-C).

Best move = most-visited root child (standard MCTS; robust against Q noise).

### 4.5 Tree reuse (tree.cpp)

`advance(move)`: find the root child matching `move`; **copy its subtree into a
fresh arena** (iterative BFS), making it the new root; drop the old arena.

Copying (rather than re-rooting in place) is the simple option: it compacts
memory, keeps child-contiguity invariants, and naturally frees the 30-odd
discarded siblings' subtrees. It happens once per game move, off the hot path.
If the played move has no node (never explored), start with a fresh root.

### 4.6 Live statistics & training export

- `stats()`: reads root atomics + walks the PV (most-visited path). No lock;
  values may be a few simulations stale, which is fine for display.
- `tree_snapshot()`: called after search completes. Walks the root subtree
  (configurable depth/min-visits filter), reconstructs FENs by replaying moves
  from the root board, returns flat numpy arrays via pybind. This is the
  training-data hook for learning value/policy from search statistics.

---

## 5. pybind Boundary (bindings.cpp)

Exposed as `chessengine._mcts`; `engine.py` wraps it. Rules of the boundary:

- Data crosses in coarse units only: FEN strings in, stats structs / numpy
  arrays out. No per-node Python objects, no Python callbacks from workers.
- `search()`/`start()` release the GIL for their whole duration.
- The (later) PyTorch evaluator is the single deliberate GIL-touching point,
  once per batch, from one thread.

---

## 6. Testing Strategy

| Layer | Test |
|---|---|
| C++ movegen | perft suite vs known counts + differential vs python-chess |
| MCTS, sequential | fixed seed ⇒ bit-identical results; hand-checkable tiny trees; mate-in-1/2 puzzles solved |
| MCTS, parallel | same puzzles solved; root visit distribution statistically close to sequential; ThreadSanitizer build in CI |
| Eval queue | batching correctness under contention; partial-batch timeout |
| Tree reuse | search → advance → subtree stats preserved; equivalent to fresh search on same position (statistically) |
| Python | Game rules delegation, engine API round-trips, CLI rendering (snapshot tests) |

---

## 7. Milestones

1. ✅ **M1 — plumbing:** repo/build (uv + scikit-build-core + CMake + pybind11),
   `Game`, minimal CLI (board display, human vs human).
2. ✅ **M2 — C++ core:** board, movegen, perft gate green.
3. ✅ **M3 — sequential MCTS:** arena tree, PUCT, material eval through the batch
   queue, blocking `search()`; engine plays legal, sensible chess in the CLI.
4. ✅ **M4 — parallelism:** worker pool, virtual loss, atomics, TSan clean;
   live stats in the CLI; convergence stop.
5. ✅ **M5 — tree reuse + training export:** `advance()`, `tree_snapshot()`.
6. **M6 — learned evaluation:** PyTorch batch evaluator, policy priors,
   self-play data generation.

---

## 8. Open Points (deliberately deferred)

- Exact centipawn↔win-probability mapping constant (pick once, keep everywhere).
- NN input encoding (planes) — decided at M6, doesn't affect earlier layers.
- Transpositions are ignored (tree, not DAG) — standard for AlphaZero-style
  engines; revisit only if profiling shows it matters.
- Pondering (thinking on opponent's time) — the design allows it (`start`/`stop`
  + tree reuse) but it's out of scope for now.

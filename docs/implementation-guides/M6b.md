# Implementation Guide: M6b — Encoding, Python Bridge, Network

Implements `docs/design/DESIGN-M6.md` §3, §5, §6. **Prerequisite: M6a is
merged** (Evaluator interface v2 with `EvalRequest`, atomic priors, root
noise). After this slice the engine can play through the CLI with a
randomly initialized PyTorch net.

Ground rules:

- Build after C++ changes: `uv sync --reinstall-package chessengine`
- Test: `uv run pytest`
- torch is an **optional** dependency; nothing in the default install may
  import it at module import time. All torch tests use
  `pytest.importorskip("torch")`.

Files:

```
cpp/eval/encode.h/.cpp            # new: planes + move index
CMakeLists.txt                    # add encode.cpp to BOTH targets
cpp/bindings.cpp                  # encode exports, PyEvaluator, Engine ctor
python/chessengine/engine.py      # EngineConfig.evaluator, Engine.close()
python/chessengine/eval/__init__.py, torch_eval.py   # new package
python/chessengine/ui/cli.py      # --evaluator / --net flags
pyproject.toml                    # [dependency-groups] train = ["torch>=2.3"]
tests/python/test_encoding.py     # new
tests/python/test_py_evaluator.py # new
tests/python/test_torch_eval.py   # new (importorskip torch)
```

---

## Step 1 — `cpp/eval/encode.h/.cpp`

```cpp
#pragma once
#include <span>
#include "core/board.h"

namespace eval {

inline constexpr int PLANES = 19;
inline constexpr int POLICY_SIZE = 73 * 64;  // 4672

// Writes PLANES*64 floats (plane-major, then square 0..63 in canonical
// orientation) for `board` into `out`. out.size() must be PLANES*64.
void encode_planes(const core::Board& board, std::span<float> out);

// Index of `move` in the flat policy output, for `board`'s side to move.
int move_index(const core::Board& board, core::Move move);

}  // namespace eval
```

### Canonicalization (DESIGN-M6.md §3.1)

`int canon(int sq, core::Color stm) { return stm == core::WHITE ? sq : sq ^ 56; }`
(`sq ^ 56` mirrors the rank, keeps the file — squares are a1=0..h8=63, same
as python-chess). "Us" = `board.side_to_move()`, "them" = `opponent(us)`.

### Planes (§3.2)

Zero the buffer, then:

| plane | fill |
|---|---|
| 0–5 | for each set bit `sq` of `board.pieces(us, pt)`, `out[pt*64 + canon(sq)] = 1` (pt = PAWN..KING) |
| 6–11 | same for `them`, planes `6+pt` |
| 12–15 | constant 1.0 planes when `can_castle(us, true)`, `can_castle(us, false)`, `can_castle(them, true)`, `can_castle(them, false)` |
| 16 | if `board.ep_square() >= 0`: `out[16*64 + canon(ep)] = 1` |
| 17 | constant `std::min(1.0f, board.halfmove_clock() / 100.0f)` |
| 18 | constant 1.0 |

Iterate bitboards with the pattern already used in `movegen.cpp`
(`std::countr_zero` + clear-lowest-bit).

### Move index (§3.3)

Everything below uses canonical squares: `from = canon(move.from(), stm)`,
`to = canon(move.to(), stm)`; `dr = rank_of(to) - rank_of(from)`,
`df = file_of(to) - file_of(from)`. Result: `move_type * 64 + from`.

1. **Underpromotion** (`move.promotion()` is KNIGHT/BISHOP/ROOK):
   `move_type = 64 + (df + 1) * 3 + piece_idx` with piece_idx N=0, B=1,
   R=2. (`df` is −1/0/+1; queen promotions fall through to case 3.)
2. **Knight move** (`(|dr|,|df|)` is `(1,2)` or `(2,1)`): look up
   `(dr, df)` in a fixed 8-entry table
   `{{+1,+2},{+2,+1},{+2,-1},{+1,-2},{-1,-2},{-2,-1},{-2,+1},{-1,+2}}`;
   `move_type = 56 + table_index`.
3. **Queen move** (everything else, including castling and queen
   promotions): `dist = std::max(std::abs(dr), std::abs(df))`, direction
   from `(sign(dr), sign(df))` looked up in the fixed order
   N `(+1,0)`, NE `(+1,+1)`, E `(0,+1)`, SE `(-1,+1)`, S `(-1,0)`,
   SW `(-1,-1)`, W `(0,-1)`, NW `(+1,-1)`;
   `move_type = dir * 7 + (dist - 1)`.

The exact table orders are **free choices** — the net learns whatever
mapping C++ defines, and C++ is the only encoder. What the tests must
guarantee is: every legal move maps into `[0, 4672)`, distinct moves of a
position map to distinct indices, and the mapping is mirror-consistent.
Throw `std::logic_error` if no case matches (unreachable for legal moves).

Add `encode.cpp` to **both** targets in `CMakeLists.txt` (`_mcts` and
`search_stress`).

## Step 2 — pybind exports

In `bindings.cpp` (add `#include <pybind11/numpy.h>` and
`"eval/encode.h"`):

```cpp
m.attr("PLANES") = eval::PLANES;
m.attr("POLICY_SIZE") = eval::POLICY_SIZE;

m.def("encode_planes", [](const std::string& fen) {
    py::array_t<float> out({eval::PLANES, 8, 8});
    eval::encode_planes(Board(fen),
                        {out.mutable_data(), size_t(eval::PLANES) * 64});
    return out;
});

m.def("move_indices", [](const std::string& fen,
                         const std::vector<std::string>& ucis) {
    Board board(fen);
    std::vector<int> out;
    for (const auto& u : ucis) out.push_back(eval::move_index(board, Move::from_uci(u)));
    return out;
});
```

## Step 3 — `PyEvaluator` bridge (bindings.cpp)

Callback contract (document it in a comment):
`callback(planes: float32 [N,19,8,8]) -> (values: float32 [N], logits: float32 [N,4672])`,
values are win probabilities in [0,1] for the side to move of each
(canonically encoded) position.

```cpp
class PyEvaluator : public eval::Evaluator {
public:
    explicit PyEvaluator(py::object callback) : callback_(std::move(callback)) {}

    void evaluate(std::span<const eval::EvalRequest> batch) override {
        const size_t n = batch.size();
        // 1. Encode outside the GIL.
        input_.resize(n * eval::PLANES * 64);
        for (size_t i = 0; i < n; ++i)
            eval::encode_planes(*batch[i].board,
                                {&input_[i * eval::PLANES * 64], eval::PLANES * 64ul});

        // 2. One GIL scope per batch: call Python, copy plain floats out.
        //    Every py:: object must be constructed AND destroyed inside
        //    this block (destructors need the GIL too).
        values_.assign(n, 0.5f);
        gathered_.assign(n, {});
        {
            py::gil_scoped_acquire gil;
            try {
                py::array_t<float> planes({py::ssize_t(n), py::ssize_t(eval::PLANES),
                                           py::ssize_t(8), py::ssize_t(8)},
                                          input_.data());  // copies
                py::tuple out = callback_(planes);
                auto values = py::cast<py::array_t<float>>(out[0]);
                auto logits = py::cast<py::array_t<float>>(out[1]);
                auto v = values.unchecked<1>();
                auto l = logits.unchecked<2>();
                for (size_t i = 0; i < n; ++i) {
                    values_[i] = v(py::ssize_t(i));
                    gathered_[i].reserve(batch[i].moves.size());
                    for (core::Move mv : batch[i].moves)
                        gathered_[i].push_back(
                            l(py::ssize_t(i), eval::move_index(*batch[i].board, mv)));
                }
            } catch (const py::error_already_set& e) {
                // A broken callback must not take down the search thread:
                // report once, fall back to neutral values + uniform priors.
                py::print("PyEvaluator callback failed:", e.what());
                for (auto& g : gathered_) g.clear();
            }
        }

        // 3. Outside the GIL: softmax per position over the legal subset.
        for (size_t i = 0; i < n; ++i) {
            *batch[i].value_out = values_[i];
            write_priors(batch[i], gathered_[i]);   // softmax, or uniform if empty
        }
    }
    ...
};
```

`write_priors`: if `gathered` is empty but `moves` is not → uniform;
otherwise numerically stable softmax (subtract max, exp, normalize).
Scratch vectors are members — this runs on the single evaluator thread.

**Engine wiring:** the composition root becomes

```cpp
class Engine {
public:
    Engine(const mcts::SearchConfig& config, py::object evaluator)
        : evaluator_(evaluator.is_none()
                         ? std::unique_ptr<eval::Evaluator>(std::make_unique<eval::MaterialEvaluator>())
                         : std::make_unique<PyEvaluator>(std::move(evaluator))),
          search_(config, *evaluator_) {}
    ...
    std::unique_ptr<eval::Evaluator> evaluator_;  // must outlive search_
    mcts::Search search_;
};
```

Bind as `.def(py::init<const mcts::SearchConfig&, py::object>(), py::arg("config"), py::arg("evaluator") = py::none())`.

**GIL shutdown rule (important, add as a comment):** destroying the Engine
joins the evaluator thread, which may need the GIL for its current batch.
If the destructor runs while a search is active and the caller holds the
GIL, that deadlocks. Rule: a search must be stopped (`stop()` — which
releases the GIL) before the Engine is dropped. Enforce Python-side in
step 5.

## Step 4 — `torch_eval.py` (new package `python/chessengine/eval/`)

`pyproject.toml`: add `train = ["torch>=2.3"]` under `[dependency-groups]`
(install with `uv sync --group train`). Import torch **inside**
`torch_eval.py` only — `chessengine/eval/__init__.py` stays empty.

`PolicyValueNet(nn.Module)`, per DESIGN-M6.md §6:

- `__init__(self, blocks: int = 4, filters: int = 64)`; store both on
  `self` for checkpointing.
- Trunk: `Conv2d(19, filters, 3, padding=1)` + `BatchNorm2d` + ReLU, then
  `blocks` residual blocks (two `Conv2d(filters, filters, 3, padding=1)` +
  BN each; ReLU after adding the skip).
- Policy head: `Conv2d(filters, 73, 1)` → `flatten(1)` → raw logits
  `[N, 4672]`. **No softmax.** (Flattening `[N,73,8,8]` yields index
  `plane*64 + square` — exactly `move_index`.)
- Value head: `Conv2d(filters, 8, 1)` + BN + ReLU → flatten →
  `Linear(512, 128)` + ReLU → `Linear(128, 1)` → `sigmoid` → `[N]`.

```python
class TorchEvaluator:
    """Engine evaluator callback backed by PolicyValueNet.

    __call__(planes: np.ndarray [N,19,8,8]) -> (values [N], logits [N,4672]),
    both float32 numpy. Runs on the C++ evaluator thread under the GIL.
    """

    def __init__(self, checkpoint: str | Path | None = None,
                 blocks: int = 4, filters: int = 64): ...
    def __call__(self, planes): ...   # torch.no_grad, model.eval()
    def save(self, path): ...         # {"blocks", "filters", "state_dict"}
```

- `torch.set_num_threads(1)` in `__init__` (self-play parallelizes over
  processes instead).
- `checkpoint` given → load arch params + weights from the file (so the
  file alone reconstructs the model); `None` → random init with the given
  size.
- `__call__`: `torch.from_numpy(planes)` → forward →
  `values.numpy().astype(np.float32)`, `logits.numpy().astype(np.float32)`.

## Step 5 — `engine.py` and CLI

`engine.py`:

- `EngineConfig` gains `evaluator: Callable | None = None` — any callable
  with the step-3 contract; `None` = built-in material evaluator. Pass it
  as the second `_mcts.Engine` argument.
- Add `Engine.close()`: `if self._engine.running(): self._engine.stop()`
  — then drop the reference (`self._engine = None`). Call it from
  `__del__` (guard with `getattr`/try) and support
  `__enter__`/`__exit__`. Docstring: required when a Python evaluator is
  set, per the GIL shutdown rule.

`ui/cli.py`: add `--evaluator {material,torch}` (default material) and
`--net PATH` (implies torch; `None` = random weights). For torch, build
`TorchEvaluator` (import inside the branch — material mode must not import
torch) and use larger structural defaults unless overridden:
`batch_size=64`, `workers=2`.

## Step 6 — Tests

`tests/python/test_encoding.py` — no torch needed:

1. **Reference planes:** implement an independent pure-Python encoder with
   `python-chess` (mirror via `chess.square_mirror`, same plane table) and
   compare `_mcts.encode_planes(fen)` against it over a diverse FEN list —
   reuse the corpus in `tests/python/test_perft.py` plus positions with ep
   rights, partial castling rights, high halfmove clocks, black to move.
2. **Move-index validity:** for each corpus FEN: indices of all legal
   moves (`_mcts.legal_moves`) are within `[0, 4672)` and pairwise
   distinct.
3. **Mirror consistency:** for each FEN, build the color-flipped position
   with `chess.Board(fen).mirror()`. Planes must be identical, and each
   legal move's index must equal its mirrored counterpart's
   (`chess.square_mirror` both squares, same promotion).
4. **Spot checks:** hand-computed indices for a handful of moves from the
   start position (e.g. `e2e4`, `g1f3`), castling `e1g1`, an ep capture,
   and one underpromotion — for both colors.

`tests/python/test_py_evaluator.py` — no torch, fake callbacks:

1. **Uniform logits ⇒ uniform priors:** callback returns
   `values=0.5, logits=zeros` → search runs; `tree_view` root-child priors
   uniform, sum ≈ 1.
2. **Crafted logits:** callback puts a large logit (e.g. +10) at
   `_mcts.move_indices(fen, ["e2e4"])[0]` for every position → after a
   short search, `e2e4` has prior ≈ 1 at the root and is the most-visited
   root child.
3. **Values flow:** callback returning constant 0.9, search with
   `workers=1, max_simulations=1` → `result.root_value == pytest.approx(0.1, abs=1e-3)`.
   Why exactly 0.1: the single simulation evaluates one root child; its
   0.9 is the win probability for the side to move *there* (the
   opponent), so the root's side-to-move value is 1 − 0.9. (Do not test
   with many simulations — the per-ply perspective flips make the
   root value tree-shape-dependent and the assertion flaky.)
4. **Liveness:** callback that `time.sleep(0.01)`s each batch; `start()` a
   search, poll `engine.stats()` a few times from the main thread, then
   `stop()` — no deadlock (this is the GIL-rules test).
5. **Broken callback:** callback that raises → search still terminates
   (neutral values), no crash.
6. **batch shape:** callback asserts `planes.shape == (N, 19, 8, 8)`,
   `planes.dtype == np.float32`, `N <= batch_size`.

`tests/python/test_torch_eval.py` — `pytest.importorskip("torch")`:

1. Net forward: shapes `[N]`/`[N, 4672]`, values in (0, 1), finite logits.
2. save → load round-trip reproduces outputs bit-exactly on a fixed input
   (and reconstructs a non-default `blocks`/`filters`).
3. End-to-end: `Engine(EngineConfig(evaluator=TorchEvaluator(), workers=1, batch_size=16))`,
   200 simulations from the start position → returns a legal best move.

## Definition of done

- [ ] Full suite green with the default (torch-less) install; torch tests
      green under `uv sync --group train`.
- [ ] M6a pin test and all M6a policy tests untouched and green (material
      path is bit-identical — `PyEvaluator` is additive).
- [ ] A full CLI game plays: `uv run chessengine --evaluator torch`
      (random weights; expect weak but legal play).
- [ ] TSan target still builds and runs clean (encode.cpp is in it).

## Pitfalls

- Every `py::` object must be created **and** destroyed while holding the
  GIL — keep them scoped inside the `gil_scoped_acquire` block; only plain
  `float` vectors leave it.
- `Engine` member order: the evaluator must be declared before `search_`
  (the queue thread uses it until `Search`'s destructor completes).
- Don't `import torch` at package import time anywhere reachable from
  `import chessengine`.
- `py::array_t` may hand you non-contiguous arrays from a careless
  callback; `unchecked` requires the exact ndim, and `py::cast` to
  `py::array_t<float>` does not force a copy — use
  `py::array_t<float, py::array::c_style | py::array::forcecast>` for both
  outputs to normalize dtype/layout.
- The canonical-square XOR is `^ 56` (rank mirror), not `^ 63` (that would
  also mirror files).

# chessengine

An educational chess engine built around Monte Carlo Tree Search, in the
spirit of AlphaZero: a parallel C++ search core behind pybind11, a Python
layer for game rules (python-chess) and the terminal UI, and pluggable
batched evaluation — a material heuristic today, a PyTorch policy/value
network later. Design and rationale live in [DESIGN.md](DESIGN.md).

## Quick start

```sh
uv sync              # builds the C++ extension and installs everything
uv run chessengine   # play against the engine in the terminal
uv run pytest        # run the test suite
```

```
uv run chessengine [--color white|black] [--time SECONDS] [--workers N] [--human]
```

While the engine thinks, the CLI shows live search statistics (simulations,
nodes, evaluation, principal variation). Ctrl-C during a search plays the
best move found so far.

## How it works

- **Game state (Python):** python-chess validates moves and is the source of
  truth for the game being played.
- **Search (C++):** MCTS with PUCT selection over a fixed-capacity node arena
  (32-byte atomic nodes, contiguous child blocks). All workers share one
  tree; virtual loss spreads them across branches. `workers=1` is the fully
  sequential, deterministic reference implementation.
- **Evaluation (pluggable, always batched):** workers park in-flight
  simulations on their virtual loss and submit leaves to an evaluation queue
  in batches; a dedicated evaluator thread scores them together. The material
  heuristic and the future neural network use the identical interface.
- **Own rules in C++:** the search core has its own board and move generator
  (bitboards, copy-make), cross-validated against python-chess by exact perft
  counts (313M nodes) and differential tests.
- **Termination:** fixed time, simulation cap, convergence (evaluation stalled
  AND best move stable over a window), or user interruption.

Throughput on an M-series laptop with the material evaluator: ~640k
simulations/s sequential, ~1.4M simulations/s with 8 workers. The parallel
tree updates are ThreadSanitizer-clean (see below).

## Milestones

- [x] M1 — build plumbing, `Game` wrapper, CLI (human vs human)
- [x] M2 — C++ board + move generator, perft gate vs python-chess
- [x] M3 — sequential MCTS, batched material evaluation, engine plays
- [x] M4 — tree parallelism (virtual loss), async search, live CLI stats
- [x] M5 — tree reuse across moves, training-data export
- [ ] M6 — PyTorch policy/value evaluator, self-play training

## Development

- Layout: Python package in `python/chessengine/`, C++ in `cpp/`
  (`core/` rules, `mcts/` search, `eval/` evaluators), tests in `tests/`.
- After changing C++ sources: `uv sync --reinstall-package chessengine`.
- Deep perft tests: `uv run pytest -m slow`.
- ThreadSanitizer stress run:

  ```sh
  cmake -S . -B build/tsan -DCHESSENGINE_TSAN=ON \
        -Dpybind11_DIR=$(uv run --with pybind11 python -m pybind11 --cmakedir)
  cmake --build build/tsan --target search_stress
  ./build/tsan/search_stress 8 30000
  ```

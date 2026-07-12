# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A chess engine based on Monte Carlo Tree Search (MCTS), designed in the spirit of AlphaGo/AlphaZero: a fast parallel C++ search core, a Python layer for game rules and visualization, and optional neural-network evaluation via PyTorch. The long-term goal is self-play training: the engine exports its internal search statistics so winning probabilities can be learned for arbitrary positions.

## Architecture

Four components with strict separation of concerns:

### 1. Chess program (Python)
- Built on the **python-chess** library.
- Source of truth for the actual game: tracks the board position and whose turn it is.
- Validates moves (legality, check, castling, en passant, promotion, draw rules).
- Exposes state to the GUI and to the search engine; applies moves coming from either.

### 2. GUI (Python, CLI-based)
- Separate module from the chess program — visualization must be easily replaceable later (e.g., by a web or graphical frontend).
- Pulls state from the chess program, renders the board in the terminal, and accepts move input from the user.
- **Also visualizes live search-engine state:** nodes visited, current score/evaluation of the position, and similar search statistics, updated while the engine is thinking.
- Contains no game logic and no engine logic.

### 3. Search engine (C++ with pybind11 bindings)
- MCTS starting from a position + side-to-move handed over from the chess program.
- **Own move generation:** the C++ engine has its own board representation and move generator, used during search. Calling back into Python from C++ hot paths is a no-go. The resulting rules duplication (C++ vs python-chess) is accepted and cross-validated via perft tests against python-chess.
- **Tree reuse:** accepts an optional existing search tree, so when playing a sequence of moves the relevant subtree from the previous search is carried over instead of starting cold.
- **Termination:** returns after a configurable fixed time, on convergence (see below), or on user interruption.
- **Convergence criterion:** search stops early when **both** hold: the root evaluation has stalled (no meaningful change over a window of simulations) **and** the best move has been stable over that window. Window size and stall threshold are configurable. (MCTS natively yields a win probability; a standard logistic mapping converts to centipawns for display.)
- **Parallelism: tree parallelism with virtual loss.** All worker threads share a single tree; visit counts and value sums are updated atomically; virtual loss spreads concurrent threads across different branches. The number of workers is configurable; **1 worker (fully sequential) must always work** and serves as the sanity-check reference for the parallel implementation.
- **Memory:** node storage designed for efficient access patterns (cache-friendly layout, avoid pointer-chasing and false sharing between threads).
- **Live statistics:** exposes search progress (nodes visited, current best score, principal variation, …) to Python while running, for display in the GUI.

### 4. Evaluation functions (pluggable)
- Leaf evaluation happens **always in batched mode**: search workers enqueue leaf positions (parking on the pending result via virtual loss), an evaluator drains the queue and returns values in batches. Early in the search batches are small; as the tree widens, many positions can be evaluated together.
- The evaluator interface is pluggable behind a common batch API:
  - **Initial evaluator: cheap material heuristic** (implemented in C++, but still driven through the batch interface so the plumbing is identical).
  - **Later: learned AlphaGo-style policy/value network** (Python / PyTorch), swapped in without changing the search.
- **Training loop support:** the search engine exports its internal state (per-node visit counts, value estimates, principal variation) back to Python so it can be used as training data for learning winning probabilities from positions.

## Data Flow

```
GUI ──(user move)──► Chess program ──(position, turn, [old tree])──► Search engine
 ▲   ▲                    │  ▲                                            │
 │   └──(live search stats: nodes visited, score, PV)─────────────────────┤
 └──────(state)───────────┘  └───────────(best move, search stats)────────┘
                                                                          │
                              Batched evaluator ◄──(leaf batches)─────────┤
                              (material heuristic now,                    │
                               PyTorch policy/value later) ──(values)────►│
                                                                          │
                              Training data (tree statistics) ◄───────────┘
```

## Design Principles

- **This is an educational project.** Runtime performance is crucial, but readability and simplicity always win over squeezing out the last percent. Prefer the clear implementation; optimize only where profiling shows it matters.
- Visualization is strictly decoupled from game logic and search — swapping the GUI must not touch other components.
- Sequential execution (1 worker) is the correctness reference; parallel runs are validated against it.
- No Python callbacks from C++ hot paths; all C++↔Python data exchange happens in coarse, batched form.
- Evaluation is always batched and behind a pluggable interface, so heuristics and learned models are interchangeable.
- Every performance-relevant knob (worker count, search time, convergence criteria, batch size) is configurable.
- CPU-only for now; the design should not preclude GPU/batched NN inference later.

## Toolchain

- **Python:** managed with `uv`; game rules via `python-chess`; PyTorch for learned evaluation.
- **C++ extension:** CMake + pybind11.
- Perft tests cross-validate the C++ move generator against python-chess.

## Setup and Development

- `uv sync` — builds the C++ extension (scikit-build-core + CMake + pybind11) and installs all dependencies.
- `uv run pytest` — run the test suite.
- `uv run chessengine` — play in the terminal.
- After changing C++ sources: `uv sync --reinstall-package chessengine` to rebuild.
- Layout: Python package in `python/chessengine/`, C++ sources in `cpp/`, tests in `tests/`.

## Status

Design accepted — see `DESIGN.md` for the concrete design (interfaces, data
structures, threading model) and the milestone plan. Implementation in progress.

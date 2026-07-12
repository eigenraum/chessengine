# chessengine

Educational chess engine based on Monte Carlo Tree Search: Python game layer
(python-chess) and CLI, parallel C++ search core (pybind11), pluggable batched
evaluation (material heuristic now, AlphaZero-style PyTorch net later).

See `DESIGN.md` for the design and milestone plan.

## Quick start

```sh
uv sync              # builds the C++ extension and installs everything
uv run chessengine   # play in the terminal
uv run pytest        # run the tests
```

After changing C++ sources, rebuild with `uv sync --reinstall-package chessengine`.

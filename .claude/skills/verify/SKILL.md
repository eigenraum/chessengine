---
name: verify
description: Build, run, and drive this project's CLI to verify engine/UI changes end-to-end.
---

# Verifying chessengine changes

## Build

- `uv sync --reinstall-package chessengine` — rebuild the C++ extension after touching `cpp/`.
- Ignore clangd "file not found" diagnostics on `cpp/`; the CMake build is the truth.

## Drive the CLI (the runtime surface)

The CLI reads moves from stdin, so piping works:

```sh
printf 'e2e4\nd2d4\nquit\n' | uv run chessengine --time 0.5
```

- Accepts SAN (`e4`, `Nf3`) and UCI (`e2e4`); `moves`, `new`, `help`, `quit` are commands.
- Useful flags: `--time SECONDS`, `--workers N`, `--no-converge`, `--human`, `--color black`.
- Live "thinking..." lines use `\r`; in a pipe they concatenate — grep for `thinking\|engine:`.
- Tree reuse shows up as `nodes` >> `sims` in the engine result line, and as a
  nonzero node count in the first live stats line of the second engine move.

## Gates

- `uv run pytest` (add `-m slow` for the deep perft gate).
- After touching parallel search: TSan stress run (README § Development):
  `cmake --build build/tsan --target search_stress && ./build/tsan/search_stress 8 30000`
  (configure once with `-DCHESSENGINE_TSAN=ON`, see README for the full command).

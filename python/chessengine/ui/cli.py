"""CLI frontend: renders the board and reads moves from the terminal.

Pure visualization/input layer — holds no game state and no rules. Everything
it shows comes from Game accessors; every move goes through Game.push(), which
validates it. Swapping this module for another frontend must not touch the
rest of the code (DESIGN.md section 2.3).
"""

from __future__ import annotations

import argparse
import time

from chessengine.engine import Engine, EngineConfig, SearchLimits
from chessengine.game import Game, IllegalMoveError

# Letter symbols (from Game.piece_map) -> unicode chess glyphs.
_GLYPHS = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

_HELP = """\
Enter moves in SAN (Nf3, e4, O-O) or UCI (g1f3, e2e4) notation.
Commands: moves  - list legal moves
          new    - start a new game
          help   - show this help
          quit   - exit"""


def render_board(game: Game) -> str:
    """Render the position as a multi-line string, White at the bottom."""
    pieces = game.piece_map()
    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = [str(rank + 1)]
        for file in range(8):
            symbol = pieces.get(rank * 8 + file)
            row.append(_GLYPHS[symbol] if symbol else "·")
        row.append(str(rank + 1))
        lines.append(" ".join(row))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def render_status(game: Game) -> str:
    outcome = game.outcome()
    if outcome is not None:
        result = outcome.result()
        reason = outcome.termination.name.lower().replace("_", " ")
        return f"Game over: {result} ({reason})"
    side = "White" if game.turn else "Black"
    return f"{side} to move"


def _render_history(game: Game) -> str:
    sans = game.san_history()
    parts = []
    for i in range(0, len(sans), 2):
        moves = " ".join(sans[i : i + 2])
        parts.append(f"{i // 2 + 1}. {moves}")
    return " ".join(parts)


def render_search_result(result, engine_san: str, white_cp: int) -> str:
    pv = " ".join(result.pv[:6])
    return (
        f"engine: {engine_san}  ({result.simulations:,} sims, "
        f"{result.nodes:,} nodes, {result.elapsed_ms} ms, "
        f"eval {white_cp:+d}cp for White, stopped: {result.stop_reason})\n"
        f"        pv: {pv}"
    )


def _engine_move(engine: Engine, game: Game, limits: SearchLimits) -> str:
    """Search in the background, showing live stats while the engine thinks."""
    engine_is_white = game.turn  # before pushing
    engine.set_position(game.fen())
    engine.start(limits)
    try:
        while engine.running():
            stats = engine.stats()
            white_cp = stats.root_cp if engine_is_white else -stats.root_cp
            line = (
                f"thinking...  {stats.simulations:,} sims  {stats.nodes:,} nodes  "
                f"eval {white_cp:+d}cp  pv {' '.join(stats.pv[:5])}"
            )
            print(f"\r{line[:98]:<98}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass  # fall through: stop() returns the best move found so far
    result = engine.stop()
    print("\r" + " " * 98 + "\r", end="")

    game.push(result.best_move)
    white_cp = result.root_cp if engine_is_white else -result.root_cp
    return render_search_result(result, game.san_history()[-1], white_cp)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chessengine", description="MCTS chess engine, terminal frontend"
    )
    parser.add_argument("--human", action="store_true", help="two human players, no engine")
    parser.add_argument(
        "--color", choices=["white", "black"], default="white", help="your side (default white)"
    )
    parser.add_argument(
        "--time", type=float, default=3.0, metavar="SECONDS", help="engine think time per move"
    )
    parser.add_argument("--workers", type=int, default=1, help="search worker threads")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    engine = None if args.human else Engine(EngineConfig(workers=args.workers))
    limits = SearchLimits(max_time_ms=int(args.time * 1000))
    human_is_white = args.color == "white"

    game = Game()
    print("chessengine — " + ("human vs human" if args.human else f"you play {args.color}"))
    print(_HELP)
    while True:
        print()
        print(render_board(game))
        history = _render_history(game)
        if history:
            print(history)
        print(render_status(game))
        if game.is_over():
            command = input("new game? [y/N] ").strip().lower()
            if command == "y":
                game = Game()
                continue
            return

        if engine is not None and game.turn != human_is_white:
            print("engine is thinking...")
            print(_engine_move(engine, game, limits))
            continue

        try:
            entry = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not entry:
            continue
        if entry in ("quit", "exit"):
            return
        if entry == "new":
            game = Game()
            continue
        if entry == "help":
            print(_HELP)
            continue
        if entry == "moves":
            print(" ".join(sorted(move.uci() for move in game.legal_moves())))
            continue

        try:
            game.push(entry)
        except IllegalMoveError as err:
            print(f"{err} — type 'moves' for legal moves, 'help' for help")


if __name__ == "__main__":
    main()

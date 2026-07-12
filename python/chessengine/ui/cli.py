"""CLI frontend: renders the board and reads moves from the terminal.

Pure visualization/input layer — holds no game state and no rules. Everything
it shows comes from Game accessors; every move goes through Game.push(), which
validates it. Swapping this module for another frontend must not touch the
rest of the code (DESIGN.md section 2.3).
"""

from __future__ import annotations

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


def main() -> None:
    game = Game()
    print("chessengine — human vs human (engine arrives with milestone M3)")
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

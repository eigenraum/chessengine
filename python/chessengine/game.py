"""Chess program: source of truth for the game state, backed by python-chess.

Validates moves and tracks the position. Knows nothing about the GUI or the
search engine; both pull state from here.
"""

from __future__ import annotations

import chess


class IllegalMoveError(ValueError):
    """Raised when a move is not legal in the current position."""


class Game:
    def __init__(self, fen: str | None = None) -> None:
        self._board = chess.Board(fen) if fen else chess.Board()
        self._san_history: list[str] = []

    @property
    def turn(self) -> chess.Color:
        """Side to move (chess.WHITE or chess.BLACK)."""
        return self._board.turn

    def fen(self) -> str:
        return self._board.fen()

    def legal_moves(self) -> list[chess.Move]:
        return list(self._board.legal_moves)

    def push(self, move: str | chess.Move) -> chess.Move:
        """Apply a move given as SAN ("Nf3"), UCI ("g1f3"), or a chess.Move.

        Raises IllegalMoveError if the move cannot be parsed or is not legal.
        Returns the applied move.
        """
        if isinstance(move, chess.Move):
            parsed = move
        else:
            parsed = self._parse(move)
        if parsed not in self._board.legal_moves:
            raise IllegalMoveError(f"illegal move: {move}")
        self._san_history.append(self._board.san(parsed))
        self._board.push(parsed)
        return parsed

    def _parse(self, move: str) -> chess.Move:
        try:
            return self._board.parse_san(move)
        except ValueError:
            pass
        try:
            return self._board.parse_uci(move)
        except ValueError:
            raise IllegalMoveError(f"illegal move: {move}") from None

    def rewind(self, ply: int) -> None:
        """Rewind to the position after `ply` half-moves (0 = starting position).

        Only moves played through this Game can be taken back; raises
        ValueError if ply is out of range.
        """
        if not 0 <= ply <= len(self._san_history):
            raise ValueError(f"ply out of range: {ply}")
        while len(self._board.move_stack) > ply:
            self._board.pop()
        del self._san_history[ply:]

    def outcome(self) -> chess.Outcome | None:
        """Game result, or None while the game is still running."""
        return self._board.outcome(claim_draw=True)

    def is_over(self) -> bool:
        return self.outcome() is not None

    def san_history(self) -> list[str]:
        return list(self._san_history)

    def piece_map(self) -> dict[int, str]:
        """Square index (0=a1 .. 63=h8) -> piece symbol ("P", "n", ...).

        This is the render-friendly view of the position the GUI consumes.
        """
        return {sq: piece.symbol() for sq, piece in self._board.piece_map().items()}

    def last_move(self) -> chess.Move | None:
        return self._board.peek() if self._board.move_stack else None

    def check_square(self) -> int | None:
        """Square of the checked king (side to move), or None if not in check."""
        return self._board.king(self._board.turn) if self._board.is_check() else None

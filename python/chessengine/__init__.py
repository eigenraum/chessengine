"""Educational MCTS chess engine.

Layers (see DESIGN.md): ui -> game -> engine (pybind) -> C++ mcts/core.
"""

from chessengine.game import Game, IllegalMoveError

__all__ = ["Game", "IllegalMoveError"]

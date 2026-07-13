"""Educational MCTS chess engine.

Layers (see DESIGN.md): ui -> game -> engine (pybind) -> C++ mcts/core.
"""

from chessengine.engine import (
    Engine,
    EngineConfig,
    SearchLimits,
    SearchResult,
    SearchStats,
    TreeSnapshot,
)
from chessengine.game import Game, IllegalMoveError

__all__ = [
    "Engine",
    "EngineConfig",
    "Game",
    "IllegalMoveError",
    "SearchLimits",
    "SearchResult",
    "SearchStats",
    "TreeSnapshot",
]

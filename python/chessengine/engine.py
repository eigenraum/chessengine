"""Pythonic facade over the C++ search core (chessengine._mcts).

Owns the config defaults and converts between plain dataclasses on the Python
side and the pybind structs on the C++ side. Positions go in as FEN strings;
results come back as stats structs — nothing finer-grained crosses the
boundary (DESIGN.md section 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chessengine import _mcts


@dataclass
class EngineConfig:
    workers: int = 1  # 1 = fully sequential reference mode
    batch_size: int = 8  # max leaf evaluations per batch
    c_puct: float = 1.5  # PUCT exploration constant
    virtual_loss: int = 1
    max_nodes: int = 1 << 22  # search-tree arena capacity (~128 MB)
    seed: int = 0


@dataclass
class SearchLimits:
    max_time_ms: int = 5000
    max_simulations: int = -1  # -1 = no simulation limit
    # Converged = over the last `convergence_window` simulations the root
    # evaluation drifted less than the cp threshold AND the best move did not
    # change. window <= 0 disables early stopping.
    convergence_window: int = 2000
    convergence_cp_threshold: int = 5


@dataclass
class SearchStats:
    simulations: int = 0
    nodes: int = 0
    root_value: float = 0.5  # win probability, side-to-move's view
    root_cp: int = 0  # the same, as centipawns
    best_move: str = ""  # UCI; empty if the position has no legal moves
    pv: list[str] = field(default_factory=list)
    elapsed_ms: int = 0


@dataclass
class SearchResult(SearchStats):
    stop_reason: str = ""


def _stats_kwargs(cxx) -> dict:
    return dict(
        simulations=cxx.simulations,
        nodes=cxx.nodes,
        root_value=cxx.root_value,
        root_cp=cxx.root_cp,
        best_move=cxx.best_move,
        pv=list(cxx.pv),
        elapsed_ms=cxx.elapsed_ms,
    )


class Engine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        config = config or EngineConfig()
        cxx_config = _mcts.SearchConfig()
        cxx_config.workers = config.workers
        cxx_config.batch_size = config.batch_size
        cxx_config.c_puct = config.c_puct
        cxx_config.virtual_loss = config.virtual_loss
        cxx_config.max_nodes = config.max_nodes
        cxx_config.seed = config.seed
        self._engine = _mcts.Engine(cxx_config)

    def set_position(self, fen: str) -> None:
        """Start a fresh search tree from this position."""
        self._engine.set_position(fen)

    def search(self, limits: SearchLimits | None = None) -> SearchResult:
        """Run a blocking search; returns the best move and search statistics."""
        result = self._engine.search(self._cxx_limits(limits))
        return SearchResult(**_stats_kwargs(result), stop_reason=result.stop_reason)

    def start(self, limits: SearchLimits | None = None) -> None:
        """Start a search in the background; poll stats(), finish with stop()."""
        self._engine.start(self._cxx_limits(limits))

    def stop(self) -> SearchResult:
        """Interrupt a running search (no-op if already done) and collect its result."""
        result = self._engine.stop()
        return SearchResult(**_stats_kwargs(result), stop_reason=result.stop_reason)

    def running(self) -> bool:
        return self._engine.running()

    def stats(self) -> SearchStats:
        """Current search statistics; cheap, safe to call any time."""
        return SearchStats(**_stats_kwargs(self._engine.stats()))

    def request_stop(self) -> None:
        """Ask a running search to stop; it returns its result promptly."""
        self._engine.request_stop()

    @staticmethod
    def _cxx_limits(limits: SearchLimits | None):
        limits = limits or SearchLimits()
        cxx_limits = _mcts.SearchLimits()
        cxx_limits.max_time_ms = limits.max_time_ms
        cxx_limits.max_simulations = limits.max_simulations
        cxx_limits.convergence_window = limits.convergence_window
        cxx_limits.convergence_cp_threshold = limits.convergence_cp_threshold
        return cxx_limits

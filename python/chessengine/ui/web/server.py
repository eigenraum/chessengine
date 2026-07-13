"""Browser frontend: FastAPI server owning one Game + Engine session.

Fills the same role as ui/cli.py, in the browser (DESIGN-VISU.md). The server
is the single writer to Game and Engine; browsers send commands over REST and
receive live state and search statistics over one WebSocket (/ws/events).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import threading
import time
import webbrowser
from pathlib import Path

import chess
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chessengine.engine import Engine, EngineConfig, SearchLimits, SearchStats
from chessengine.game import Game, IllegalMoveError

STATIC_DIR = Path(__file__).parent / "static"
STATS_INTERVAL_S = 0.25  # ~4 Hz stats stream (DESIGN-VISU.md section 5.1)
TREE_EVERY_TICKS = 4  # tree snapshots at ~1 Hz (section 5.2)
TREE_MAX_NODES = 20_000  # node budget per snapshot
CP_SCALE = 400.0  # centipawn<->win-prob logistic scale (cpp/eval/evaluator.h)


class Session:
    """One game + one engine; all mutations go through this object.

    All methods run on the server's single event loop, so game/engine access
    needs no locking; the only concurrency is the C++ search behind
    engine.start(), which is observed via the lock-free stats()/running().
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        limits: SearchLimits | None = None,
    ) -> None:
        self.game = Game()
        self.config = config or EngineConfig()
        self.engine = Engine(self.config)
        self.engine.set_position(self.game.fen())
        self.limits = limits or SearchLimits()
        self._clients: set[WebSocket] = set()
        self._search_task: asyncio.Task | None = None
        self._play_on_stop = True
        # Explicit flag, not _search_task.done(): the task itself broadcasts
        # the post-search state and must already read searching == False then.
        self._searching = False

    # ---- state -----------------------------------------------------------

    @property
    def searching(self) -> bool:
        return self._searching

    def state(self) -> dict:
        outcome = self.game.outcome()
        last = self.game.last_move()
        check = self.game.check_square()
        return {
            "type": "state",
            "fen": self.game.fen(),
            "turn": "w" if self.game.turn == chess.WHITE else "b",
            "legal_moves": [m.uci() for m in self.game.legal_moves()],
            "history": self.game.san_history(),
            "outcome": None
            if outcome is None
            else {
                "result": outcome.result(),
                "termination": outcome.termination.name.lower(),
            },
            "last_move": last.uci() if last else None,
            "check_square": chess.square_name(check) if check is not None else None,
            "searching": self.searching,
        }

    def _stats_event(self, stats: SearchStats, sims_per_s: float, nodes_per_s: float) -> dict:
        # root_value/root_cp are from the side to move's view; the eval bar
        # wants white's view.
        white_to_move = self.game.turn == chess.WHITE
        return {
            "type": "stats",
            "simulations": stats.simulations,
            "nodes": stats.nodes,
            "white_win_prob": stats.root_value if white_to_move else 1.0 - stats.root_value,
            "white_cp": stats.root_cp if white_to_move else -stats.root_cp,
            "best_move": stats.best_move,
            "pv": stats.pv,
            "pv_san": self._pv_san(stats.pv),
            "elapsed_ms": stats.elapsed_ms,
            "sims_per_s": round(sims_per_s),
            "nodes_per_s": round(nodes_per_s),
        }

    def tree_event(
        self, root_path: list[str] | None = None, max_nodes: int = TREE_MAX_NODES
    ) -> dict:
        """Bounded snapshot of the (sub)tree at root_path; safe while searching."""
        view = self.engine.tree_view(max_nodes=max_nodes, root_path=root_path)
        return {
            "type": "tree",
            "turn": "w" if self.game.turn == chess.WHITE else "b",
            "root_path": root_path or [],
            "parent": view.parent,
            "move": view.move,
            "visits": view.visits,
            "q": view.q,
            "prior": view.prior,
            "children_total": view.children_total,
        }

    def config_event(self) -> dict:
        """Both parameter groups (§4.3) plus the eval-mapping constant."""
        return {
            "type": "config",
            "limits": dataclasses.asdict(self.limits),
            "structural": dataclasses.asdict(self.config),
            "cp_scale": CP_SCALE,
            # a running search still uses the limits it started with
            "searching": self.searching,
        }

    def tree_fens(self, paths: list[list[str]]) -> list[str | None]:
        """FENs for tree nodes (L3 thumbnails, §5.2): replay each move path
        from the current position. None for paths that don't replay (the tree
        is racy while searching; the client just skips those thumbnails)."""
        fens: list[str | None] = []
        for path in paths:
            board = chess.Board(self.game.fen())
            try:
                for uci in path:
                    board.push(board.parse_uci(uci))
                fens.append(board.fen())
            except ValueError:
                fens.append(None)
        return fens

    def _pv_san(self, pv: list[str]) -> list[str]:
        """PV as SAN for display; stats are racy, so stop at the first move
        that does not parse."""
        board = chess.Board(self.game.fen())
        san = []
        for uci in pv:
            try:
                move = board.parse_uci(uci)
            except ValueError:
                break
            san.append(board.san(move))
            board.push(move)
        return san

    # ---- broadcasting ----------------------------------------------------

    async def broadcast(self, event: dict) -> None:
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def broadcast_state(self) -> None:
        await self.broadcast(self.state())

    # ---- commands --------------------------------------------------------

    async def play_move(self, move: str) -> None:
        """A human move: validate, apply, keep the engine's tree warm."""
        if self.searching:
            raise HTTPException(409, "search in progress")
        applied = self.game.push(move)  # raises IllegalMoveError
        self.engine.advance(applied.uci())
        await self.broadcast_state()

    async def new_game(self, fen: str | None = None) -> None:
        await self.stop_search(play_move=False)
        try:
            self.game = Game(fen)
        except ValueError as exc:
            raise HTTPException(400, f"invalid FEN: {exc}") from None
        self.engine.set_position(self.game.fen())
        await self.broadcast_state()

    async def set_position(self, fen: str) -> None:
        """Apply an edited position (§3.2): validated, tree dropped."""
        await self.stop_search(play_move=False)
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise HTTPException(400, f"invalid FEN: {exc}") from None
        if not board.is_valid():
            reasons = [flag.name.lower() for flag in chess.Status if flag & board.status()]
            raise HTTPException(400, f"invalid position: {', '.join(reasons)}")
        self.game = Game(board.fen())
        self.engine.set_position(self.game.fen())
        await self.broadcast_state()
        await self.broadcast(self.tree_event())

    async def goto_ply(self, ply: int) -> None:
        """Takeback (§3.3): rewind the game; the engine tree is dropped."""
        await self.stop_search(play_move=False)
        try:
            self.game.rewind(ply)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        self.engine.set_position(self.game.fen())
        await self.broadcast_state()
        await self.broadcast(self.tree_event())

    async def goto_path(self, path: list[str]) -> None:
        """Click-to-explore (§4.2): play the clicked node's moves into the
        real game; each hop keeps the matching subtree (tree reuse)."""
        await self.stop_search(play_move=False)
        board = chess.Board(self.game.fen())  # validate before mutating
        try:
            for uci in path:
                board.push(board.parse_uci(uci))
        except ValueError as exc:
            raise HTTPException(400, f"path not playable: {exc}") from None
        for uci in path:
            applied = self.game.push(uci)
            self.engine.advance(applied.uci())
        await self.broadcast_state()
        await self.broadcast(self.tree_event())

    async def update_config(
        self, limits: dict | None = None, structural: dict | None = None
    ) -> None:
        """The two §4.3 lifecycles: limits apply at the next search start;
        structural changes rebuild the engine (tree dropped)."""
        try:
            if limits:
                self.limits = dataclasses.replace(self.limits, **limits)
            if structural:
                self.config = dataclasses.replace(self.config, **structural)
        except TypeError as exc:
            raise HTTPException(400, f"unknown parameter: {exc}") from None
        if structural:
            await self.stop_search(play_move=False)
            self.engine = Engine(self.config)
            self.engine.set_position(self.game.fen())
            await self.broadcast(self.tree_event())
        await self.broadcast(self.config_event())

    async def start_search(self) -> None:
        """The Move! button. While a search runs it acts as Stop (accepted
        decision: interrupt and play the best-so-far move)."""
        if self.searching:
            await self.stop_search()
            return
        if self.game.is_over():
            raise HTTPException(409, "game is over")
        self._play_on_stop = True
        self.engine.start(self.limits)
        self._searching = True
        self._search_task = asyncio.create_task(self._run_search())
        await self.broadcast_state()

    async def stop_search(self, play_move: bool = True) -> None:
        """Interrupt the search; by default the best-so-far move is played."""
        task = self._search_task
        if task is None or task.done():
            return
        self._play_on_stop = play_move
        self.engine.request_stop()
        await task

    async def _run_search(self) -> None:
        """Stream stats while the engine runs, then play its move."""
        last_sims = last_nodes = 0
        last_t = time.monotonic()
        tick = 0
        while True:  # do-while: even the shortest search emits one stats event
            stats = self.engine.stats()
            now = time.monotonic()
            dt = max(now - last_t, 1e-3)
            await self.broadcast(
                self._stats_event(
                    stats,
                    sims_per_s=(stats.simulations - last_sims) / dt,
                    nodes_per_s=(stats.nodes - last_nodes) / dt,
                )
            )
            last_sims, last_nodes, last_t = stats.simulations, stats.nodes, now
            if tick % TREE_EVERY_TICKS == 0:
                await self.broadcast(self.tree_event())
            tick += 1
            if not self.engine.running():
                break
            await asyncio.sleep(STATS_INTERVAL_S)

        result = self.engine.stop()
        # final tree view of the decided position, before advance() re-roots it
        await self.broadcast(self.tree_event())
        played = None
        try:
            if self._play_on_stop and result.best_move and not self.game.is_over():
                move = self.game.push(result.best_move)
                self.engine.advance(move.uci())
                played = move.uci()
        finally:
            self._searching = False  # before broadcasting: state must say idle
        rate = result.simulations / max(result.elapsed_ms / 1000, 1e-3)
        node_rate = result.nodes / max(result.elapsed_ms / 1000, 1e-3)
        await self.broadcast(
            {
                **self._stats_event(result, rate, node_rate),
                "type": "search_end",
                "stop_reason": result.stop_reason,
                "played_move": played,
            }
        )
        await self.broadcast_state()


class MoveRequest(BaseModel):
    uci: str


class NewGameRequest(BaseModel):
    fen: str | None = None


class PositionRequest(BaseModel):
    fen: str


class GotoRequest(BaseModel):
    ply: int | None = None  # takeback: rewind to after `ply` half-moves
    path: list[str] | None = None  # explore: play these moves from here


class ConfigRequest(BaseModel):
    limits: dict | None = None
    structural: dict | None = None


class TreeDetailRequest(BaseModel):
    root_path: list[str]
    max_nodes: int = TREE_MAX_NODES


class TreeFensRequest(BaseModel):
    paths: list[list[str]]


def create_app(session: Session | None = None) -> FastAPI:
    session = session or Session()
    app = FastAPI(title="chessengine")
    app.state.session = session

    @app.get("/api/state")
    async def get_state() -> dict:
        return session.state()

    @app.post("/api/move")
    async def post_move(req: MoveRequest) -> dict:
        try:
            await session.play_move(req.uci)
        except IllegalMoveError as exc:
            raise HTTPException(400, str(exc)) from None
        return session.state()

    @app.post("/api/new")
    async def post_new(req: NewGameRequest | None = None) -> dict:
        await session.new_game(req.fen if req else None)
        return session.state()

    @app.post("/api/position")
    async def post_position(req: PositionRequest) -> dict:
        await session.set_position(req.fen)
        return session.state()

    @app.post("/api/goto")
    async def post_goto(req: GotoRequest) -> dict:
        if (req.ply is None) == (req.path is None):
            raise HTTPException(400, "exactly one of ply or path required")
        if req.ply is not None:
            await session.goto_ply(req.ply)
        else:
            await session.goto_path(req.path or [])
        return session.state()

    @app.get("/api/config")
    async def get_config() -> dict:
        return session.config_event()

    @app.put("/api/config")
    async def put_config(req: ConfigRequest) -> dict:
        await session.update_config(req.limits, req.structural)
        return session.config_event()

    @app.get("/api/tree")
    async def get_tree() -> dict:
        return session.tree_event()

    @app.post("/api/tree/detail")
    async def post_tree_detail(req: TreeDetailRequest) -> dict:
        return session.tree_event(req.root_path, req.max_nodes)

    @app.post("/api/tree/fens")
    async def post_tree_fens(req: TreeFensRequest) -> dict:
        return {"fens": session.tree_fens(req.paths)}

    @app.post("/api/search/start")
    async def post_search_start() -> dict:
        await session.start_search()
        return session.state()

    @app.post("/api/search/stop")
    async def post_search_stop() -> dict:
        await session.stop_search()
        return session.state()

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        session._clients.add(ws)
        try:
            await ws.send_json(session.state())
            while True:  # server-push only; read to detect disconnect
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            session._clients.discard(ws)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="chessengine web frontend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=EngineConfig.workers, help="search worker threads")
    parser.add_argument("--time-ms", type=int, default=SearchLimits.max_time_ms, help="search time per move")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    session = Session(
        config=EngineConfig(workers=args.workers),
        limits=SearchLimits(max_time_ms=args.time_ms),
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"chessengine web UI: {url}")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, [url]).start()
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

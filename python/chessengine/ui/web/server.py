"""Browser frontend: FastAPI server owning one Game + Engine session.

Fills the same role as ui/cli.py, in the browser (DESIGN-VISU.md). The server
is the single writer to Game and Engine; browsers send commands over REST and
receive live state and search statistics over one WebSocket (/ws/events).
"""

from __future__ import annotations

import argparse
import asyncio
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
        self.engine = Engine(config)
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
            if not self.engine.running():
                break
            await asyncio.sleep(STATS_INTERVAL_S)

        result = self.engine.stop()
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

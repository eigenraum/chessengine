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
from chessengine.ui.web.shard_view import load_game

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
        self._analysing = False  # infinite analysis (§10.3): never auto-plays
        # one eval per searched position, keyed by ply (§11.2): the sparkline
        self.eval_history: list[dict] = []

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
            "analysing": self._analysing,
            "eval_history": self.eval_history,
        }

    def _record_eval(self, stats_event: dict) -> None:
        """§11.2: remember the eval of the just-searched position for the
        sparkline. The event is built pre-push, so the ply is the current
        history length; a re-search of the same ply overwrites its entry."""
        entry = {
            "ply": len(self.game.san_history()),
            "white_win_prob": stats_event["white_win_prob"],
            "white_cp": stats_event["white_cp"],
        }
        self.eval_history = [e for e in self.eval_history if e["ply"] != entry["ply"]]
        self.eval_history.append(entry)
        self.eval_history.sort(key=lambda e: e["ply"])

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
            # the position this snapshot is rooted at — the client sends it
            # back with /api/tree/fens so paths replay against the right base
            # even when the game has moved on (§10.1)
            "fen": self.game.fen(),
            "root_path": root_path or [],
            "parent": view.parent,
            "move": view.move,
            "visits": view.visits,
            "q": view.q,
            "prior": view.prior,
            "children_total": view.children_total,
        }

    def config_event(self) -> dict:
        """Both parameter groups (§4.3) plus the eval-mapping constant.

        `structural` is built by hand, not dataclasses.asdict(self.config):
        EngineConfig.evaluator is a live callable (or None) — not
        JSON-serializable, and asdict() would deepcopy it (a full model
        copy, for a TorchEvaluator) just to build a response. It's a CLI
        startup choice, not adjustable through this API, so only a
        human-readable summary is exposed here.
        """
        return {
            "type": "config",
            "limits": dataclasses.asdict(self.limits),
            "structural": {
                "workers": self.config.workers,
                "batch_size": self.config.batch_size,
                "max_nodes": self.config.max_nodes,
                "seed": self.config.seed,
                "evaluator": "torch" if self.config.evaluator is not None else "material",
            },
            "cp_scale": CP_SCALE,
            # a running search still uses the limits it started with
            "searching": self.searching,
        }

    def tree_fens(self, paths: list[list[str]], fen: str | None = None) -> dict:
        """FENs (plus the last move as SAN) for tree nodes (L3 thumbnails,
        §5.2): replay each move path from `fen` — the base position of the
        client's tree snapshot (§10.1; defaults to the current game). None
        for paths that don't replay (the tree is racy while searching; the
        client just skips those thumbnails)."""
        base = fen or self.game.fen()
        try:
            chess.Board(base)
        except ValueError as exc:
            raise HTTPException(400, f"invalid FEN: {exc}") from None
        fens: list[str | None] = []
        sans: list[str | None] = []
        for path in paths:
            board = chess.Board(base)
            try:
                san = None
                for uci in path:
                    move = board.parse_uci(uci)
                    san = board.san(move)
                    board.push(move)
                fens.append(board.fen())
                sans.append(san)
            except ValueError:
                fens.append(None)
                sans.append(None)
        return {"fens": fens, "sans": sans}

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
        self.eval_history = []
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
        self.eval_history = []
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
        # evals past the rewind point describe abandoned positions (§11.2)
        self.eval_history = [e for e in self.eval_history if e["ply"] <= ply]
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

    async def start_search(self, analyse: bool = False) -> None:
        """The Move! button. While a search runs it acts as Stop (accepted
        decision: interrupt and play the best-so-far move). With analyse
        (§10.3) the search has no time limit or convergence stop and never
        plays a move — the tree just keeps growing until stopped."""
        if self.searching:
            await self.stop_search()
            return
        if self.game.is_over():
            raise HTTPException(409, "game is over")
        limits = self.limits
        if analyse:
            limits = dataclasses.replace(
                limits, max_time_ms=0, max_simulations=-1, convergence_window=0
            )
        self._play_on_stop = not analyse
        self._analysing = analyse
        self.engine.start(limits)
        self._searching = True
        self._search_task = asyncio.create_task(self._run_search())
        await self.broadcast_state()

    async def step_search(self, steps: int) -> None:
        """§10.4: exactly `steps` MCTS descents (select → expand → evaluate →
        backprop), accumulated on the existing tree; no move is played."""
        if self.searching:
            raise HTTPException(409, "search in progress")
        if self.game.is_over():
            raise HTTPException(409, "game is over")
        if steps < 1:
            raise HTTPException(400, "steps must be >= 1")
        self._play_on_stop = False
        self._analysing = False
        self.engine.start(
            dataclasses.replace(
                self.limits, max_time_ms=0, max_simulations=steps, convergence_window=0
            )
        )
        self._searching = True
        self._search_task = asyncio.create_task(self._run_search())
        await self.broadcast_state()

    async def play_best(self) -> None:
        """§10.3: play the engine's current best move — the most-visited root
        child of the live tree, which tracks the game position by
        construction. Stops a running (analysis) search first, quietly."""
        await self.stop_search(play_move=False)
        if self.game.is_over():
            raise HTTPException(409, "game is over")
        view = self.engine.tree_view(max_nodes=64)  # best-first: the top child is in
        best, best_visits = None, 0
        for parent, move, visits in zip(view.parent, view.move, view.visits):
            if parent == 0 and visits > best_visits:
                best, best_visits = move, visits
        if best is None:
            raise HTTPException(409, "no analysis for this position yet")
        applied = self.game.push(best)
        self.engine.advance(applied.uci())
        await self.broadcast_state()
        await self.broadcast(self.tree_event())

    async def stop_search(self, play_move: bool | None = None) -> None:
        """Interrupt the search. play_move=None keeps the behavior the search
        was started with (play the best-so-far move for Move! searches,
        nothing for analysis/step searches)."""
        task = self._search_task
        if task is None or task.done():
            return
        if play_move is not None:
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
        # build the final stats event BEFORE the move is pushed: pv/eval are
        # rooted in the searched position, and pushing flips game.turn (which
        # would invert white_win_prob and make the pv unparseable as SAN)
        rate = result.simulations / max(result.elapsed_ms / 1000, 1e-3)
        node_rate = result.nodes / max(result.elapsed_ms / 1000, 1e-3)
        end_event = self._stats_event(result, rate, node_rate)
        if result.simulations > 0:
            self._record_eval(end_event)
        played = None
        try:
            if self._play_on_stop and result.best_move and not self.game.is_over():
                move = self.game.push(result.best_move)
                self.engine.advance(move.uci())
                played = move.uci()
        finally:
            self._searching = False  # before broadcasting: state must say idle
            self._analysing = False
        await self.broadcast(
            {
                **end_event,
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
    fen: str | None = None  # base position of the client's tree (§10.1)


class SearchStartRequest(BaseModel):
    analyse: bool = False  # infinite analysis (§10.3)


class StepRequest(BaseModel):
    steps: int = 1  # MCTS descents per click (§10.4)


class ShardLoadRequest(BaseModel):
    path: str  # server-side path to a self-play .npz shard


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
        return session.tree_fens(req.paths, req.fen)

    @app.post("/api/selfplay/load")
    async def post_selfplay_load(req: ShardLoadRequest) -> dict:
        """Read-only self-play shard viewer (docs/readme-training.md): loads
        an .npz written by chessengine-selfplay and reconstructs its game
        for the "Self-play" tab. Stateless — unrelated to the live
        Game/Engine session; the client keeps the loaded game client-side."""
        try:
            return load_game(req.path)
        except Exception as exc:
            raise HTTPException(400, f"could not read shard: {exc}") from None

    @app.post("/api/search/start")
    async def post_search_start(req: SearchStartRequest | None = None) -> dict:
        await session.start_search(analyse=bool(req and req.analyse))
        return session.state()

    @app.post("/api/search/step")
    async def post_search_step(req: StepRequest | None = None) -> dict:
        await session.step_search(req.steps if req else 1)
        return session.state()

    @app.post("/api/play/best")
    async def post_play_best() -> dict:
        await session.play_best()
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

    # No-build frontend: without Cache-Control browsers cache the JS modules
    # heuristically and can keep running stale code after an update. no-cache
    # still allows conditional requests (304s via the StaticFiles ETag).
    @app.middleware("http")
    async def no_heuristic_caching(request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="chessengine web frontend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=None, help="search worker threads")
    parser.add_argument("--batch-size", type=int, default=None, help="evaluator batch size")
    parser.add_argument("--time-ms", type=int, default=SearchLimits.max_time_ms, help="search time per move")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--evaluator",
        choices=["material", "torch"],
        default="material",
        help="leaf evaluator: cheap material heuristic, or a PyTorch policy/value net",
    )
    parser.add_argument(
        "--net",
        metavar="PATH",
        help="checkpoint for --evaluator torch (default: random-initialized weights)",
    )
    args = parser.parse_args()

    # Mirrors ui/cli.py's _make_engine: a torch net wants a bigger batch and
    # a couple of workers to fill it; the material default (workers=1,
    # batch_size=8) stays the baseline (DESIGN-M6.md section 5).
    if args.evaluator == "torch":
        # Imported here, not at module level: material mode must not import torch.
        from chessengine.eval.torch_eval import TorchEvaluator

        evaluator = TorchEvaluator(checkpoint=args.net)
        workers = args.workers if args.workers is not None else 2
        batch_size = args.batch_size if args.batch_size is not None else 64
    else:
        evaluator = None
        workers = args.workers if args.workers is not None else EngineConfig.workers
        batch_size = args.batch_size if args.batch_size is not None else EngineConfig.batch_size

    session = Session(
        config=EngineConfig(workers=workers, batch_size=batch_size, evaluator=evaluator),
        limits=SearchLimits(max_time_ms=args.time_ms),
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"chessengine web UI: {url}")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, [url]).start()
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

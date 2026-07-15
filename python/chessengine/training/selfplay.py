"""Self-play game generation (DESIGN-M6.md section 7.1/7.2).

Per game: search each move with root Dirichlet noise, record `tree_snapshot`
rows (root + interior), pick the played move by visit-count temperature,
advance the tree, and once the game ends fill in `outcome` (final result
from each row's side-to-move perspective) before writing one `.npz` shard.

    chessengine-selfplay --net best.pt --out data/gen3 --games 500 --jobs 8
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chessengine import _mcts
from chessengine.engine import Engine, EngineConfig, SearchLimits
from chessengine.game import Game
from chessengine.training.dataset import Row, save_game_shard


def white_result_from_outcome(outcome) -> float:
    """1.0 white win, 0.0 black win, 0.5 draw (None outcome = ply-cap adjudication)."""
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner else 0.0


def row_outcome(fen: str, white_result: float) -> float:
    """The `outcome` field for a row: `white_result` from that row's
    side-to-move perspective (parsed from the FEN's side-to-move field)."""
    side = fen.split()[1]
    return white_result if side == "w" else 1.0 - white_result


def pick_move(
    moves: list[str], child_visits: list[int], ply: int, temp_plies: int,
    rng: np.random.Generator,
) -> str:
    """ply < temp_plies: sample proportional to visit counts (tau=1); else argmax."""
    visits = np.asarray(child_visits, dtype=np.float64)
    if ply < temp_plies:
        return moves[rng.choice(len(moves), p=visits / visits.sum())]
    return moves[int(np.argmax(visits))]


@dataclass
class SelfPlayConfig:
    sims: int
    workers: int
    batch_size: int
    temp_plies: int
    noise_eps: float
    dirichlet_alpha: float
    snapshot_min_visits: int
    snapshot_max_depth: int
    max_plies: int


def play_one_game(
    evaluator, net_path: str, game_index: int, seed: int,
    config: SelfPlayConfig, out_dir: Path,
) -> Path:
    game_seed = seed + game_index
    game = Game()
    rows: list[Row] = []
    rng = np.random.default_rng(game_seed)

    with Engine(
        EngineConfig(
            evaluator=evaluator, workers=config.workers,
            batch_size=config.batch_size, seed=game_seed,
        )
    ) as engine:
        engine.set_position(game.fen())
        ply = 0
        while game.outcome() is None and ply < config.max_plies:
            engine.search(
                SearchLimits(
                    max_time_ms=0, max_simulations=config.sims, convergence_window=0,
                    root_noise_eps=config.noise_eps,
                    root_dirichlet_alpha=config.dirichlet_alpha,
                )
            )
            snap = engine.tree_snapshot(
                min_visits=config.snapshot_min_visits, max_depth=config.snapshot_max_depth
            )
            for i in range(len(snap)):
                moves_i = snap.moves[i]
                if not moves_i:
                    continue  # terminal / never-expanded node: no policy target
                visits_i = np.asarray(snap.child_visits[i], dtype=np.float64)
                probs = (visits_i / visits_i.sum()).astype(np.float32).tolist()
                rows.append(
                    Row(
                        fen=snap.fens[i],
                        policy_index=_mcts.move_indices(snap.fens[i], moves_i),
                        policy_prob=probs,
                        search_value=float(snap.values[i]),
                        visit_count=int(snap.visit_counts[i]),
                        is_root=(i == 0),
                    )
                )
            move = pick_move(snap.moves[0], snap.child_visits[0], ply, config.temp_plies, rng)
            game.push(move)
            engine.advance(move)
            ply += 1

    white_result = white_result_from_outcome(game.outcome())
    for row in rows:
        row.outcome = row_outcome(row.fen, white_result)

    out_path = out_dir / f"game-{int(time.time() * 1000)}-{os.getpid()}-{game_index}.npz"
    save_game_shard(
        out_path,
        rows,
        meta={
            "net": net_path,
            "sims": config.sims,
            "noise_eps": config.noise_eps,
            "dirichlet_alpha": config.dirichlet_alpha,
            "engine_version": _mcts.version(),
        },
    )
    return out_path


def _play_games_in_worker(bundle) -> list[Path]:
    net_path, game_indices, seed, config, out_dir = bundle
    # Lazy: keeps this module (and its pure helpers above) importable
    # without the `train` dependency group; only a worker that actually
    # plays games needs torch.
    from chessengine.eval.torch_eval import TorchEvaluator

    evaluator = TorchEvaluator(checkpoint=net_path)
    return [play_one_game(evaluator, net_path, gi, seed, config, out_dir) for gi in game_indices]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate self-play games")
    parser.add_argument("--net", required=True, type=Path, help="current-best net checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="output directory for .npz shards")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--temp-plies", type=int, default=30)
    parser.add_argument("--noise-eps", type=float, default=0.25)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--snapshot-min-visits", type=int, default=8)
    parser.add_argument("--snapshot-max-depth", type=int, default=30)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2, help="search worker threads per engine")
    parser.add_argument("--batch-size", type=int, default=64, help="evaluator batch size")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> list[Path]:
    """Parses argv and generates games; returns the written shard paths.

    Kept separate from main() so a caller (tests, a driver script) can use
    the return value — main() is the console-script entry point, and must
    return None: the generated wrapper does `sys.exit(main())`, and
    sys.exit() with any non-None, non-int argument prints it and exits 1.
    """
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    config = SelfPlayConfig(
        sims=args.sims, workers=args.workers, batch_size=args.batch_size,
        temp_plies=args.temp_plies, noise_eps=args.noise_eps,
        dirichlet_alpha=args.dirichlet_alpha, snapshot_min_visits=args.snapshot_min_visits,
        snapshot_max_depth=args.snapshot_max_depth, max_plies=args.max_plies,
    )
    game_indices = list(range(args.games))
    started = time.time()

    if args.jobs <= 1:
        paths = _play_games_in_worker((str(args.net), game_indices, args.seed, config, args.out))
    else:
        chunks = [game_indices[i :: args.jobs] for i in range(args.jobs)]
        bundles = [(str(args.net), chunk, args.seed, config, args.out) for chunk in chunks]
        # spawn, not fork: a forked child would inherit a half-alive C++
        # evaluator thread from this process's own Engine(s), if any.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            results = pool.map(_play_games_in_worker, bundles)
        paths = [p for sub in results for p in sub]

    elapsed = time.time() - started
    rate = len(paths) / elapsed if elapsed > 0 else float("inf")
    print(f"{len(paths)} games in {elapsed:.1f}s ({rate:.2f} games/s) -> {args.out}")
    return paths


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()

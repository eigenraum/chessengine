"""Arena: net-vs-net matches and the promotion gate (DESIGN-M6.md section 7.5).

    chessengine-arena --net-a candidate.pt --net-b best.pt
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from chessengine.eval.device import DEVICE_CHOICES
from chessengine.engine import Engine, EngineConfig, SearchLimits
from chessengine.game import Game

if TYPE_CHECKING:
    from chessengine.eval.server import EvalServer


def a_plays_white(game_index: int) -> bool:
    return game_index % 2 == 0


def pick_move_arena(
    moves: list[str], visits: list[int], ply: int, temp_plies: int, temp: float,
    rng: np.random.Generator,
) -> str:
    """ply < temp_plies: sample p ~ visits^(1/temp) (opening variety); else argmax."""
    v = np.asarray(visits, dtype=np.float64)
    if ply < temp_plies:
        w = v ** (1.0 / temp)
        return moves[rng.choice(len(moves), p=w / w.sum())]
    return moves[int(np.argmax(v))]


def score_for_a(outcome, a_is_white: bool) -> float:
    """1.0 A won, 0.0 A lost, 0.5 draw (None outcome = ply-cap adjudication)."""
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == a_is_white else 0.0


def play_game(
    white: Engine, black: Engine, sims: int, temp_plies: int, temp: float,
    max_plies: int, rng: np.random.Generator,
):
    game = Game()
    white.set_position(game.fen())
    black.set_position(game.fen())
    ply = 0
    while game.outcome() is None and ply < max_plies:
        mover, other = (white, black) if game.turn else (black, white)
        mover.search(
            SearchLimits(
                max_time_ms=0, max_simulations=sims, convergence_window=0, root_noise_eps=0.0
            )
        )
        # tree_snapshot(min_visits=1, max_depth=1): just the root's child
        # visit distribution, cheap — full-depth export isn't needed here.
        snap = mover.tree_snapshot(min_visits=1, max_depth=1)
        move = pick_move_arena(snap.moves[0], snap.child_visits[0], ply, temp_plies, temp, rng)
        game.push(move)
        # Both engines advance every move — each maintains its own tree, so
        # the non-mover's subtree reuse stays valid once it moves next.
        mover.advance(move)
        other.advance(move)
        ply += 1
    return game.outcome()


def _play_one_game_with_servers(
    server_a: "EvalServer", server_b: "EvalServer", game_index: int, seed: int,
    sims: int, workers: int, batch_size: int, temp_plies: int, temp: float, max_plies: int,
) -> float:
    """--parallel-games worker body: one EvalServer client per net for this
    game, closed when the game ends (DESIGN-GPU.md section 5.4). Each game
    gets its own seeded rng — a shared rng threaded across concurrent
    threads would race; per-game seeding is also what selfplay.py does."""
    a_white = a_plays_white(game_index)
    rng = np.random.default_rng(seed + game_index)
    client_a = server_a.client()
    client_b = server_b.client()
    try:
        with (
            Engine(
                EngineConfig(
                    evaluator=client_a, workers=workers, batch_size=batch_size,
                    seed=seed + game_index,
                )
            ) as engine_a,
            Engine(
                EngineConfig(
                    evaluator=client_b, workers=workers, batch_size=batch_size,
                    seed=seed + game_index,
                )
            ) as engine_b,
        ):
            white, black = (engine_a, engine_b) if a_white else (engine_b, engine_a)
            outcome = play_game(white, black, sims, temp_plies, temp, max_plies, rng)
        return score_for_a(outcome, a_white)
    finally:
        client_a.close()
        client_b.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Net-vs-net arena match with a promotion gate")
    parser.add_argument("--net-a", required=True, type=Path)
    parser.add_argument("--net-b", required=True, type=Path)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temp-plies", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate", type=float, default=0.55)
    parser.add_argument(
        "--device", default="cpu", choices=DEVICE_CHOICES,
        help="net device; auto picks cuda, then Apple Silicon (mps), then cpu "
        "(default cpu — see DESIGN-GPU.md section 4.3 for --jobs x GPU tradeoffs)",
    )
    parser.add_argument(
        "--parallel-games", type=int, default=1,
        help="games run concurrently, each pair of engines backed by two shared "
        "EvalServers (one per net) that coalesce batches across games "
        "(DESIGN-GPU.md section 5.4)",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict:
    """Parses argv and runs the match; returns the score summary. Kept
    separate from main() — see selfplay.run()'s docstring for why main()
    must return None."""
    args = _parse_args(argv)
    results: list[float] = []

    def _postfix() -> dict[str, int]:
        return {
            "W": sum(1 for r in results if r == 1.0),
            "D": sum(1 for r in results if r == 0.5),
            "L": sum(1 for r in results if r == 0.0),
        }

    with tqdm(total=args.games, desc="arena", unit="game") as bar:
        if args.parallel_games > 1:
            # Lazy: keeps this module importable without the `train` group;
            # only the GPU-batching path needs torch.
            from chessengine.eval.server import EvalServer

            server_a = EvalServer(checkpoint=args.net_a, device=args.device)
            server_b = EvalServer(checkpoint=args.net_b, device=args.device)
            try:
                with ThreadPoolExecutor(max_workers=args.parallel_games) as pool:
                    futures = [
                        pool.submit(
                            _play_one_game_with_servers, server_a, server_b, g, args.seed,
                            args.sims, args.workers, args.batch_size, args.temp_plies,
                            args.temp, args.max_plies,
                        )
                        for g in range(args.games)
                    ]
                    for future in as_completed(futures):
                        results.append(future.result())
                        bar.set_postfix(_postfix())
                        bar.update(1)
            finally:
                server_a.close()
                server_b.close()
        else:
            # Lazy: keeps argument parsing (and the pure functions above)
            # importable without the `train` dependency group.
            from chessengine.eval.torch_eval import TorchEvaluator

            evaluator_a = TorchEvaluator(checkpoint=args.net_a, device=args.device)
            evaluator_b = TorchEvaluator(checkpoint=args.net_b, device=args.device)
            rng = np.random.default_rng(args.seed)

            for g in range(args.games):
                a_white = a_plays_white(g)
                with (
                    Engine(
                        EngineConfig(
                            evaluator=evaluator_a, workers=args.workers,
                            batch_size=args.batch_size, seed=args.seed + g,
                        )
                    ) as engine_a,
                    Engine(
                        EngineConfig(
                            evaluator=evaluator_b, workers=args.workers,
                            batch_size=args.batch_size, seed=args.seed + g,
                        )
                    ) as engine_b,
                ):
                    white, black = (engine_a, engine_b) if a_white else (engine_b, engine_a)
                    outcome = play_game(
                        white, black, args.sims, args.temp_plies, args.temp, args.max_plies, rng
                    )
                results.append(score_for_a(outcome, a_white))
                bar.set_postfix(_postfix())
                bar.update(1)

    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.5)
    losses = sum(1 for r in results if r == 0.0)
    score = wins + 0.5 * draws
    fraction = score / args.games
    verdict = "PROMOTE" if fraction >= args.gate else "KEEP"
    print(f"A score: {score:g}/{args.games} ({fraction:.3f}) — {verdict}")
    print(f"W {wins}  D {draws}  L {losses}")
    return {
        "score": score, "fraction": fraction, "wins": wins, "draws": draws,
        "losses": losses, "promote": fraction >= args.gate,
    }


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()

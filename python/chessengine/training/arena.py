"""Arena: net-vs-net matches and the promotion gate (DESIGN-M6.md section 7.5).

    chessengine-arena --net-a candidate.pt --net-b best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chessengine.engine import Engine, EngineConfig, SearchLimits
from chessengine.game import Game


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
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict:
    """Parses argv and runs the match; returns the score summary. Kept
    separate from main() — see selfplay.run()'s docstring for why main()
    must return None."""
    args = _parse_args(argv)

    # Lazy: keeps argument parsing (and the pure functions above) importable
    # without the `train` dependency group.
    from chessengine.eval.torch_eval import TorchEvaluator

    evaluator_a = TorchEvaluator(checkpoint=args.net_a)
    evaluator_b = TorchEvaluator(checkpoint=args.net_b)
    rng = np.random.default_rng(args.seed)

    wins = draws = losses = 0
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
        result = score_for_a(outcome, a_white)
        if result == 1.0:
            wins += 1
        elif result == 0.5:
            draws += 1
        else:
            losses += 1

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

"""Automated generation loop (docs/readme-training.md's "pipeline" section,
DESIGN-M6.md section 7): self-play -> train -> arena -> promote, repeated.

    chessengine-loop --best best.pt --data data

This is exactly the four-line pipeline from the docs, run generation after
generation until interrupted (or for a fixed --generations count), with
per-generation scalars (self-play throughput, train losses, arena score)
written to TensorBoard. Progress bars for the self-play/train/arena steps
themselves come straight from those commands' own tqdm bars, since this
module drives them through their `run()` entry points (same ones tests use)
rather than reimplementing anything.

Unlike a standalone `chessengine-arena` invocation, this loop *does*
auto-promote a candidate that clears the gate (`--no-auto-promote` to keep
the manual `cp` behavior instead): a fully automated loop has no other way
for one generation's result to become the next generation's starting net.

    chessengine-train --init --out best.pt   # generation 0, if --best is missing
    chessengine-selfplay --net best.pt --out data ...
    chessengine-train --data data --in best.pt --out candidate-*.pt ...
    chessengine-arena --net-a candidate-*.pt --net-b best.pt ...
    # PROMOTE -> cp candidate-*.pt best.pt (automatic here)

A candidate that doesn't clear the gate isn't necessarily thrown away: its
arena win rate against --best still has to clear --keep-threshold (default
0.5) or its checkpoint file is deleted, so near-miss candidates stay on disk
for later inspection/continued training while clearly-worse ones don't pile
up.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from chessengine.eval.device import DEVICE_CHOICES

logger = logging.getLogger("chessengine.loop")


def decide_candidate_action(
    promote: bool, auto_promote: bool, fraction: float, keep_threshold: float
) -> str:
    """What to do with a generation's candidate checkpoint once arena has
    scored it: "promote" (clears the gate and auto-promote is on), "keep"
    (either it cleared the gate but auto-promote is off, or its win rate is
    still above --keep-threshold even though it didn't clear the gate), or
    "discard" (win rate at or below --keep-threshold — not worth the disk
    space)."""
    if promote and auto_promote:
        return "promote"
    if promote or fraction > keep_threshold:
        return "keep"
    return "discard"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated self-play -> train -> arena -> promote loop"
    )
    parser.add_argument(
        "--best", type=Path, default=Path("best.pt"),
        help="current-best checkpoint; created (random-init) if it doesn't exist yet",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data"),
        help="shard directory shared by every generation's self-play output "
        "(flat, not per-generation subdirectories: chessengine-train's --window "
        "scans it non-recursively for the most recent games by mtime)",
    )
    parser.add_argument(
        "--generations", type=int, default=0,
        help="number of generations to run; 0 = run until interrupted (default)",
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed for generation 0")
    parser.add_argument(
        "--device", default="auto", choices=DEVICE_CHOICES,
        help="net device for self-play, training, and arena alike; "
        "auto picks cuda, then Apple Silicon (mps), then cpu",
    )
    parser.add_argument(
        "--auto-promote", action="store_true", default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-auto-promote", dest="auto_promote", action="store_false",
        help="don't copy a PROMOTE-d candidate over --best; just log the verdict "
        "and keep the candidate file (every generation then self-plays with the "
        "same --best net)",
    )
    parser.add_argument(
        "--tensorboard-dir", type=Path, default=Path("runs"),
        help="directory for TensorBoard event files",
    )
    parser.add_argument(
        "--keep-threshold", type=float, default=0.5,
        help="a non-promoted candidate is deleted unless its arena win rate "
        "clears this threshold (default 0.5 = keep anything better than a coin "
        "flip against --best, even short of --gate); promoted candidates are "
        "always kept",
    )

    selfplay = parser.add_argument_group("self-play (per generation)")
    selfplay.add_argument(
        "--selfplay-games", type=int, default=100, help="self-play games per generation"
    )
    selfplay.add_argument("--selfplay-sims", type=int, default=800)
    selfplay.add_argument("--jobs", type=int, default=1, help="parallel self-play worker processes")
    selfplay.add_argument(
        "--workers", type=int, default=2, help="search worker threads per engine"
    )
    selfplay.add_argument("--batch-size", type=int, default=64, help="evaluator batch size")
    selfplay.add_argument(
        "--parallel-games", type=int, default=8,
        help="games run concurrently against one shared net (self-play and arena "
        "alike); mutually exclusive with --jobs in self-play",
    )
    selfplay.add_argument("--max-plies", type=int, default=512)

    train = parser.add_argument_group("training (per generation)")
    train.add_argument(
        "--train-steps", type=int, default=4000, help="optimizer steps per generation"
    )
    train.add_argument("--train-batch", type=int, default=256, help="training minibatch size")
    train.add_argument(
        "--window", type=int, default=5000, help="most recent N games trained on"
    )
    train.add_argument(
        "--min-visits-interior", type=int, default=32,
        help="interior rows need at least this many visits to be trained on",
    )

    arena = parser.add_argument_group("arena (per generation)")
    arena.add_argument("--arena-games", type=int, default=100, help="candidate-vs-best games")
    arena.add_argument("--arena-sims", type=int, default=400, help="simulations per arena move")
    arena.add_argument("--gate", type=float, default=0.55, help="promotion threshold")

    args = parser.parse_args(argv)
    if args.jobs > 1 and args.parallel_games > 1:
        parser.error("--jobs and --parallel-games are mutually exclusive (pick one)")
    return args


def run(argv: list[str] | None = None) -> None:
    """Parses argv and runs the loop until --generations is reached (0 = forever)
    or interrupted. Kept separate from main() — see selfplay.run()'s docstring
    for why main() must return None."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Lazy: keeps argument parsing importable without the `train` group; only
    # actually running a generation needs torch (+ tensorboard).
    from torch.utils.tensorboard import SummaryWriter

    from chessengine.training import arena as arena_mod
    from chessengine.training import selfplay as selfplay_mod
    from chessengine.training import train as train_mod

    args.data.mkdir(parents=True, exist_ok=True)
    args.best.parent.mkdir(parents=True, exist_ok=True)

    if not args.best.exists():
        logger.info("no checkpoint at %s -- creating a random-initialized generation-0 net", args.best)
        train_mod.run(["--init", "--out", str(args.best)])

    logger.info(
        "starting loop: best=%s data=%s device=%s parallel-games=%d generations=%s auto-promote=%s",
        args.best, args.data, args.device, args.parallel_games,
        args.generations if args.generations > 0 else "unlimited", args.auto_promote,
    )

    writer = SummaryWriter(str(args.tensorboard_dir))
    generation = 1
    try:
        while args.generations <= 0 or generation <= args.generations:
            logger.info("=== generation %d ===", generation)
            gen_seed = args.seed + (generation - 1) * args.selfplay_games

            t0 = time.time()
            shard_paths = selfplay_mod.run(
                [
                    "--net", str(args.best), "--out", str(args.data),
                    "--games", str(args.selfplay_games), "--sims", str(args.selfplay_sims),
                    "--jobs", str(args.jobs), "--workers", str(args.workers),
                    "--batch-size", str(args.batch_size),
                    "--parallel-games", str(args.parallel_games), "--device", args.device,
                    "--max-plies", str(args.max_plies), "--seed", str(gen_seed),
                ]
            )
            selfplay_elapsed = time.time() - t0
            logger.info(
                "self-play: %d games in %.1fs (%.2f games/s)",
                len(shard_paths), selfplay_elapsed,
                len(shard_paths) / selfplay_elapsed if selfplay_elapsed > 0 else float("inf"),
            )
            writer.add_scalar("selfplay/games", len(shard_paths), generation)
            writer.add_scalar("selfplay/elapsed_s", selfplay_elapsed, generation)

            candidate_path = args.best.parent / f"candidate-gen{generation:04d}-{int(time.time())}.pt"
            losses = train_mod.run(
                [
                    "--data", str(args.data), "--in", str(args.best), "--out", str(candidate_path),
                    "--window", str(args.window), "--steps", str(args.train_steps),
                    "--batch", str(args.train_batch),
                    "--min-visits-interior", str(args.min_visits_interior),
                    "--device", args.device, "--seed", str(gen_seed),
                ]
            )
            logger.info(
                "train: value_loss %.4f  policy_loss %.4f -> %s",
                losses["value_loss"], losses["policy_loss"], candidate_path,
            )
            writer.add_scalar("train/value_loss", losses["value_loss"], generation)
            writer.add_scalar("train/policy_loss", losses["policy_loss"], generation)

            result = arena_mod.run(
                [
                    "--net-a", str(candidate_path), "--net-b", str(args.best),
                    "--games", str(args.arena_games), "--sims", str(args.arena_sims),
                    "--workers", str(args.workers), "--batch-size", str(args.batch_size),
                    "--parallel-games", str(args.parallel_games), "--device", args.device,
                    "--gate", str(args.gate), "--seed", str(gen_seed),
                ]
            )
            logger.info(
                "arena: %.3f (%s)  W%d D%d L%d",
                result["fraction"], "PROMOTE" if result["promote"] else "KEEP",
                result["wins"], result["draws"], result["losses"],
            )
            writer.add_scalar("arena/score_fraction", result["fraction"], generation)
            writer.add_scalar("arena/wins", result["wins"], generation)
            writer.add_scalar("arena/draws", result["draws"], generation)
            writer.add_scalar("arena/losses", result["losses"], generation)

            action = decide_candidate_action(
                result["promote"], args.auto_promote, result["fraction"], args.keep_threshold
            )
            if action == "promote":
                shutil.copyfile(candidate_path, args.best)
                logger.info("promoted %s -> %s", candidate_path, args.best)
            elif action == "keep":
                logger.info(
                    "kept %s (win rate %.3f, best unchanged at %s)",
                    candidate_path, result["fraction"], args.best,
                )
            else:
                candidate_path.unlink()
                logger.info(
                    "discarded %s (win rate %.3f <= --keep-threshold %.2f)",
                    candidate_path, result["fraction"], args.keep_threshold,
                )
            writer.add_scalar("arena/promoted", float(action == "promote"), generation)
            writer.flush()

            generation += 1
    except KeyboardInterrupt:
        logger.info("interrupted after %d generation(s)", generation - 1)
    finally:
        writer.close()


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()

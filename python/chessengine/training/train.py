"""Training loop: optimize the net on a window of recent self-play games
(DESIGN-M6.md section 7.4).

    chessengine-train --data data --in best.pt --out candidate.pt

`--init` writes a fresh random-initialized checkpoint instead of training
(generation 0's "current best"):

    chessengine-train --init --out best.pt
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from chessengine.eval.device import DEVICE_CHOICES, describe_device, select_device
from chessengine.training.dataset import filter_rows, load_window, sample_batch

logger = logging.getLogger("chessengine.train")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the policy/value net on a data window")
    parser.add_argument(
        "--init", action="store_true",
        help="write a fresh random-initialized checkpoint to --out and exit",
    )
    parser.add_argument("--data", type=Path, help="self-play shard directory")
    parser.add_argument(
        "--in", dest="in_", type=Path, metavar="CKPT",
        help="checkpoint to continue training from (omit = fresh random net)",
    )
    parser.add_argument("--out", required=True, type=Path, help="checkpoint to write")
    parser.add_argument("--window", type=int, default=5000, help="most recent N games")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-root", type=float, default=1.0)
    parser.add_argument("--lambda-interior", type=float, default=0.0)
    parser.add_argument("--min-visits-interior", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--device", default="auto", choices=DEVICE_CHOICES,
        help="auto picks cuda, then Apple Silicon (mps), then cpu",
    )
    args = parser.parse_args(argv)
    if not args.init and args.data is None:
        parser.error("--data is required unless --init is given")
    return args


def run(argv: list[str] | None = None) -> dict[str, float]:
    """Parses argv and trains (or, with --init, writes a random checkpoint);
    returns the average losses ({} for --init). Kept separate from main() —
    see selfplay.run()'s docstring for why main() must return None."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Lazy: --init and plain training both need torch, but nothing above
    # this point does, keeping argument parsing importable/testable without
    # the `train` dependency group.
    import torch
    import torch.nn.functional as F

    from chessengine.eval.torch_eval import TorchEvaluator

    if args.init:
        evaluator = TorchEvaluator()
        n_params = sum(p.numel() for p in evaluator.model.parameters())
        logger.info(
            "net: %d blocks, %d filters, %d parameters",
            evaluator.model.blocks, evaluator.model.filters, n_params,
        )
        evaluator.save(args.out)
        print(f"initialized random net -> {args.out}")
        return {}

    shards = load_window(args.data, args.window)
    if not shards:
        raise SystemExit(f"no shards found in {args.data}")
    rows = filter_rows(shards, args.lambda_root, args.lambda_interior, args.min_visits_interior)
    if not rows:
        raise SystemExit("no rows survive the filter (window/min_visits_interior too strict)")
    logger.info(
        "samples: %d training rows available (%d shards, window=%d)",
        len(rows), len(shards), args.window,
    )

    device = select_device(args.device)
    logger.info("device: %s", describe_device(device))

    evaluator = TorchEvaluator(checkpoint=args.in_) if args.in_ else TorchEvaluator()
    model = evaluator.model
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "net: %d blocks, %d filters, %d parameters", model.blocks, model.filters, n_params,
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    # An "epoch" is steps_per_epoch steps: enough randomly-drawn batches to
    # cover len(rows) rows in expectation. sample_batch draws uniformly with
    # replacement (no sequential/without-replacement pass over the data), so
    # this groups steps for progress display, not a literal full pass.
    steps_per_epoch = max(1, len(rows) // args.batch)
    total_epochs = math.ceil(args.steps / steps_per_epoch)

    value_losses: list[float] = []
    policy_losses: list[float] = []
    step = 0
    with tqdm(total=total_epochs, desc="training", unit="epoch") as epoch_bar:
        for epoch in range(total_epochs):
            steps_this_epoch = min(steps_per_epoch, args.steps - step)
            with tqdm(
                total=steps_this_epoch, desc=f"epoch {epoch + 1}/{total_epochs}",
                unit="step", leave=False,
            ) as step_bar:
                for _ in range(steps_this_epoch):
                    step += 1
                    batch = sample_batch(
                        shards, args.batch, rng, args.lambda_root, args.lambda_interior,
                        args.min_visits_interior, rows=rows,
                    )
                    batch.planes = batch.planes.to(device)
                    batch.policy_index = batch.policy_index.to(device)
                    batch.policy_prob = batch.policy_prob.to(device)
                    batch.value_target = batch.value_target.to(device)
                    values, logits = model(batch.planes)
                    value_loss = F.binary_cross_entropy(values, batch.value_target)
                    log_probs = F.log_softmax(logits, dim=1)
                    # Soft-target cross-entropy: padded entries have policy_prob == 0,
                    # so they contribute nothing regardless of their (valid, non -1)
                    # padding index.
                    policy_loss = -(
                        batch.policy_prob * log_probs.gather(1, batch.policy_index)
                    ).sum(1).mean()

                    optimizer.zero_grad()
                    (value_loss + policy_loss).backward()
                    optimizer.step()

                    value_losses.append(value_loss.item())
                    policy_losses.append(policy_loss.item())
                    step_bar.set_postfix(
                        value_loss=f"{value_loss.item():.4f}", policy_loss=f"{policy_loss.item():.4f}",
                    )
                    step_bar.update(1)
                    if step % args.log_every == 0 or step == args.steps:
                        tqdm.write(
                            f"step {step}/{args.steps}  value_loss {value_loss.item():.4f}  "
                            f"policy_loss {policy_loss.item():.4f}"
                        )
            epoch_bar.update(1)

    model.eval()
    evaluator.save(args.out)
    avg_value_loss = float(np.mean(value_losses))
    avg_policy_loss = float(np.mean(policy_losses))
    print(f"avg value_loss {avg_value_loss:.4f}  avg policy_loss {avg_policy_loss:.4f}")
    print(f"saved -> {args.out}")
    return {"value_loss": avg_value_loss, "policy_loss": avg_policy_loss}


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()

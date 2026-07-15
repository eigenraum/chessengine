"""Supervised pretraining on human games — bootstrap a net before self-play.

Reads the shards written by `chessengine-pgn-import` (the standard `.npz`
format, section 7.2) and trains the same `PolicyValueNet` used everywhere
else, so the output checkpoint is a drop-in `best.pt` for the self-play loop:

    chessengine-pgn-import --pgn data_pretrain/lichess.pgn.zst --out data_pretrain/shards
    chessengine-pretrain   --data data_pretrain/shards --out best.pt

Loss is identical to the RL trainer (train.py) — BCE on the value against the
game outcome, soft-target cross-entropy on the policy against the (one-hot,
here) played move — the only real differences are that this streams a corpus
far larger than RAM (shard-buffered via dataset.stream_shard_batches) and
does real epochs (full passes) instead of with-replacement sampling. The net
architecture defaults (4 blocks, 64 filters) match TorchEvaluator's, so the
pretrained checkpoint plugs straight into self-play/arena unchanged.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from chessengine.eval.device import DEVICE_CHOICES, describe_device, select_device
from chessengine.eval.torch_eval import FILTERS_DEFAULT
from chessengine.training.dataset import stream_shard_batches

logger = logging.getLogger("chessengine.pretrain")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised pretraining on human-game shards")
    parser.add_argument(
        "--data", required=True, type=Path,
        help="directory of game-pgn-*.npz shards (from chessengine-pgn-import)",
    )
    parser.add_argument("--out", required=True, type=Path, help="checkpoint to write")
    parser.add_argument(
        "--in", dest="in_", type=Path, metavar="CKPT",
        help="checkpoint to continue from (omit = fresh random net of --blocks/--filters)",
    )
    parser.add_argument("--blocks", type=int, default=4, help="residual blocks (fresh net only)")
    parser.add_argument(
        "--filters", type=int, default=FILTERS_DEFAULT, help="conv filters (fresh net only)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="full passes over the training shards")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--buffer-shards", type=int, default=8,
        help="shards held in memory (and shuffled across) at once; the shuffle grain",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.05,
        help="fraction of shards held out for validation loss / early stopping",
    )
    parser.add_argument(
        "--patience", type=int, default=2,
        help="stop after this many epochs with no validation-loss improvement (0 = off)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0, help="cap optimizer steps per epoch (0 = whole pass)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--device", default="auto", choices=DEVICE_CHOICES,
        help="auto picks cuda, then Apple Silicon (mps), then cpu",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, float]:
    """Parse argv and pretrain; returns the final average losses. Kept
    separate from main() so callers can use the return value (main() must
    return None — see selfplay.run()'s docstring)."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Lazy: keep argument parsing importable/testable without the `train`
    # dependency group (torch), mirroring train.py.
    import torch
    import torch.nn.functional as F

    from chessengine.eval.torch_eval import TorchEvaluator

    shard_paths = sorted(Path(args.data).glob("game-pgn-*.npz"))
    if not shard_paths:
        raise SystemExit(f"no game-pgn-*.npz shards found in {args.data} (run chessengine-pgn-import first)")

    rng = np.random.default_rng(args.seed)
    # Split by shard, not by row: a shard is one contiguous block of games, so
    # holding whole shards out keeps train and val from sharing positions of
    # the same game.
    perm = rng.permutation(len(shard_paths))
    n_val = int(len(shard_paths) * args.val_frac)
    if 0 < n_val < len(shard_paths):
        val_paths = [shard_paths[i] for i in perm[:n_val]]
        train_paths = [shard_paths[i] for i in perm[n_val:]]
    else:
        val_paths, train_paths = [], shard_paths
        logger.info("too few shards (%d) for a val split — no early stopping", len(shard_paths))
    logger.info("shards: %d train, %d val", len(train_paths), len(val_paths))

    device = select_device(args.device)
    logger.info("device: %s", describe_device(device))

    evaluator = (
        TorchEvaluator(checkpoint=args.in_)
        if args.in_
        else TorchEvaluator(blocks=args.blocks, filters=args.filters)
    )
    model = evaluator.model
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("net: %d blocks, %d filters, %d parameters", model.blocks, model.filters, n_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _to_device(batch):
        batch.planes = batch.planes.to(device)
        batch.policy_index = batch.policy_index.to(device)
        batch.policy_prob = batch.policy_prob.to(device)
        batch.value_target = batch.value_target.to(device)
        return batch

    def _losses(batch):
        values, logits = model(batch.planes)
        value_loss = F.binary_cross_entropy(values, batch.value_target)
        log_probs = F.log_softmax(logits, dim=1)
        # Soft-target cross-entropy; padded entries have policy_prob == 0 and
        # contribute nothing (here every row is a single one-hot played move).
        policy_loss = -(batch.policy_prob * log_probs.gather(1, batch.policy_index)).sum(1).mean()
        return value_loss, policy_loss

    def _validate() -> tuple[float, float]:
        model.eval()
        v_losses, p_losses = [], []
        with torch.no_grad():
            for batch in stream_shard_batches(
                val_paths, args.batch, rng, buffer_shards=args.buffer_shards, shuffle=False,
            ):
                vl, pl = _losses(_to_device(batch))
                v_losses.append(vl.item())
                p_losses.append(pl.item())
        model.train()
        vv, vp = float(np.mean(v_losses)), float(np.mean(p_losses))
        return vv, vp

    model.train()
    best_val = math.inf
    epochs_without_improvement = 0
    avg_value_loss = avg_policy_loss = math.nan
    avg_val_value_loss = avg_val_policy_loss = None

    for epoch in range(args.epochs):
        value_losses, policy_losses = [], []
        step = 0
        with tqdm(desc=f"pretrain epoch {epoch + 1}/{args.epochs}", unit="step") as bar:
            for batch in stream_shard_batches(
                train_paths, args.batch, rng, buffer_shards=args.buffer_shards, shuffle=True,
            ):
                step += 1
                value_loss, policy_loss = _losses(_to_device(batch))
                optimizer.zero_grad()
                (value_loss + policy_loss).backward()
                optimizer.step()

                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                bar.update(1)
                bar.set_postfix(
                    value_loss=f"{value_loss.item():.4f}", policy_loss=f"{policy_loss.item():.4f}",
                )
                if step % args.log_every == 0:
                    tqdm.write(
                        f"epoch {epoch + 1} step {step}  value_loss {value_loss.item():.4f}  "
                        f"policy_loss {policy_loss.item():.4f}"
                    )
                if args.max_steps and step >= args.max_steps:
                    break

        avg_value_loss = float(np.mean(value_losses)) if value_losses else math.nan
        avg_policy_loss = float(np.mean(policy_losses)) if policy_losses else math.nan
        tqdm.write(
            f"epoch {epoch + 1}/{args.epochs}  train_value_loss {avg_value_loss:.4f}  "
            f"train_policy_loss {avg_policy_loss:.4f}"
        )

        # Save after every epoch so a long run is checkpointed as it goes, not
        # only at the end.
        model.eval()
        evaluator.save(args.out)
        model.train()

        if val_paths:
            avg_val_value_loss, avg_val_policy_loss = _validate()
            val_total = avg_val_value_loss + avg_val_policy_loss
            tqdm.write(
                f"epoch {epoch + 1}/{args.epochs}  val_value_loss {avg_val_value_loss:.4f}  "
                f"val_policy_loss {avg_val_policy_loss:.4f}"
            )
            if args.patience and val_total >= best_val:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    logger.info(
                        "early stopping: val loss hasn't improved for %d epochs (best %.4f)",
                        args.patience, best_val,
                    )
                    break
            else:
                best_val = min(best_val, val_total)
                epochs_without_improvement = 0

    print(f"avg value_loss {avg_value_loss:.4f}  avg policy_loss {avg_policy_loss:.4f}")
    print(f"saved -> {args.out}")
    result = {"value_loss": avg_value_loss, "policy_loss": avg_policy_loss}
    if avg_val_value_loss is not None and avg_val_policy_loss is not None:
        result["val_value_loss"] = avg_val_value_loss
        result["val_policy_loss"] = avg_val_policy_loss
    return result


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()

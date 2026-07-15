# Supervised pretraining data

Drop a Lichess PGN dump here to bootstrap the net by supervised learning on
human games *before* self-play (a warm start beats a random init). Everything
in this directory except this file and `.gitignore` is git-ignored.

## 1. Get the data

Download a standard-rated dump from <https://database.lichess.org/> — a
`.pgn.zst` file. Put it here, e.g. `data_pretrain/lichess.pgn.zst`.

You do **not** need to decompress it: `chessengine-pgn-import` streams the
`.zst` through `zstd`/`pzstd` on the fly (no multi-gigabyte temp file).
Uncompressed dumps are ~7× larger. If you prefer a file on disk anyway:

```bash
pzstd -d data_pretrain/lichess.pgn.zst      # -> data_pretrain/lichess.pgn
```

`.zst` is partially decompressable, so a cancelled download still imports.

## 2. Import → training shards

```bash
uv run --group train chessengine-pgn-import \
    --pgn data_pretrain/lichess.pgn.zst \
    --out data_pretrain/shards \
    --min-elo 1500 --games-per-shard 2000
```

Filters (all configurable): standard chess only (no variants / Chess960),
a finished result, both players rated ≥ `--min-elo`, and at least
`--min-plies` half-moves. Output is `game-pgn-*.npz` shards in the **same
format as self-play** (`dataset.save_game_shard`): one row per position, the
played move as a one-hot policy target, and the game result (from the side to
move) as the value target. Use `--max-games` to import a quick subset first.

## 3. Pretrain

```bash
uv run --group train chessengine-pretrain \
    --data data_pretrain/shards \
    --out best.pt \
    --epochs 1 --batch 256
```

Streams the shards (only `--buffer-shards` in memory at once), trains the
standard `PolicyValueNet` with the same loss as `chessengine-train`, holds
out `--val-frac` of the shards for validation / early stopping, and writes a
checkpoint after every epoch. The output is a drop-in `best.pt`: hand it to
`chessengine-selfplay --net best.pt …` or the training loop as generation 0's
current-best instead of a random init.

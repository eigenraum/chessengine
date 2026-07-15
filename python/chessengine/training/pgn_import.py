"""Turn a Lichess PGN dump into training shards for supervised pretraining.

This is the supervised-learning counterpart to selfplay.py: instead of the
engine's own searched games, the "teacher" is a corpus of human games, and
each played move is the policy target. The output is the *exact same* `.npz`
shard format (dataset.Row / save_game_shard), so pretrain.py — and, for
small sets, chessengine-train — read it with no special-casing:

    chessengine-pgn-import --pgn data_pretrain/lichess.pgn.zst --out data_pretrain/shards

Per accepted game, every mainline position becomes one row:

- policy target: the move actually played, one-hot (`policy_prob = [1.0]`) at
  its `move_index` — behavioural cloning of the human move.
- value target: the game result from that position's side-to-move
  perspective (1 win / 0.5 draw / 0 loss). Rows are marked `is_root=True`,
  so the default `lambda_root=1.0` uses this outcome directly as the value
  target (search_value is set to the same value and is otherwise unused —
  there is no search here).

`.pgn.zst` inputs are streamed through `zstd -dc` (or `pzstd`, if present),
so no multi-gigabyte decompressed file is written to disk. A plain `.pgn`
works too. Positions are stored as FENs, not planes — planes are recomputed
at training time through `_mcts.encode_planes`, guaranteeing the pretrained
net sees the identical input representation the search will (section 3.4).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import chess.pgn
from tqdm import tqdm

from chessengine import _mcts
from chessengine.training.dataset import Row, save_game_shard

logger = logging.getLogger("chessengine.pgn_import")

# Result header -> white's score. Anything else (`*`, missing) means the game
# has no usable outcome (unfinished, adjudication error) and is skipped.
_WHITE_RESULT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


@contextlib.contextmanager
def open_pgn(path: Path) -> Iterator[io.TextIOBase]:
    """A text handle over `path`, transparently decompressing `.pgn.zst` via
    an external `zstd`/`pzstd` (streamed — no temp file). Plain `.pgn` opens
    directly. `errors="replace"` tolerates the odd non-UTF-8 byte rather than
    aborting a mult-million-game import on one bad character."""
    if path.suffix == ".zst":
        tool = "pzstd" if shutil.which("pzstd") else "zstd"
        if shutil.which(tool) is None:
            raise SystemExit(
                f"{path} is zstd-compressed but neither pzstd nor zstd is on PATH; "
                "install one (brew install zstd) or decompress first (pzstd -d)"
            )
        proc = subprocess.Popen([tool, "-dcq", str(path)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        handle = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
        try:
            yield handle
        finally:
            handle.close()
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
    else:
        with open(path, encoding="utf-8", errors="replace") as handle:
            yield handle


class _GameExtractor(chess.pgn.BaseVisitor):
    """Pulls (fen, played-uci) pairs and the result out of one game while the
    parser walks it. Returning `chess.pgn.SKIP` from `end_headers` for a
    rejected game makes python-chess skip its movetext entirely — cheap
    header-level filtering without parsing millions of moves we throw away.
    Variations are skipped (Lichess mainlines have none, but be safe); a
    parse error voids the whole game."""

    def __init__(self, accept_headers) -> None:
        super().__init__()
        self._accept_headers = accept_headers

    def begin_game(self) -> None:
        self.headers: dict[str, str] = {}
        self.samples: list[tuple[str, str]] = []  # (fen before move, uci)
        self.skipped = False
        self.errored = False

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.headers[tagname] = tagvalue

    def end_headers(self):
        if not self._accept_headers(self.headers):
            self.skipped = True
            return chess.pgn.SKIP
        return None

    def begin_variation(self):
        return chess.pgn.SKIP

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        # board is the position *before* move — exactly the (position,
        # target-move) pair we want. fen() here is the per-position cost.
        self.samples.append((board.fen(), move.uci()))

    def handle_error(self, error: Exception) -> None:
        self.errored = True

    def result(self):
        return self


def accept_headers_factory(min_elo: int):
    """A header predicate: standard chess only (no variant, no custom start
    position), a decisive-or-drawn result, and both players rated >= min_elo.
    The min-plies cut needs the move count, so it is applied later, not here."""

    def accept(headers: dict[str, str]) -> bool:
        if headers.get("Result") not in _WHITE_RESULT:
            return False
        variant = headers.get("Variant", "Standard")
        if variant not in ("Standard", ""):
            return False
        # A FEN/SetUp header means a non-standard start (Chess960, odds): the
        # engine's encoder and move index assume the standard opening frame.
        if "FEN" in headers or headers.get("SetUp") == "1":
            return False
        if min_elo > 0:
            for tag in ("WhiteElo", "BlackElo"):
                try:
                    if int(headers.get(tag, "0")) < min_elo:
                        return False
                except ValueError:
                    return False  # "?" / missing rating
        return True

    return accept


def game_to_rows(headers: dict[str, str], samples: list[tuple[str, str]]) -> list[Row]:
    """(fen, uci) pairs + result headers -> shard rows (one per position)."""
    white_result = _WHITE_RESULT[headers["Result"]]
    rows: list[Row] = []
    for fen, uci in samples:
        # side-to-move perspective, same convention as row_outcome/selfplay.
        outcome = white_result if fen.split()[1] == "w" else 1.0 - white_result
        rows.append(
            Row(
                fen=fen,
                policy_index=_mcts.move_indices(fen, [uci]),
                policy_prob=[1.0],
                search_value=outcome,  # no search; == outcome, and unused for is_root rows
                visit_count=1,
                is_root=True,
                outcome=outcome,
            )
        )
    return rows


def run(argv: list[str] | None = None) -> list[Path]:
    """Parse argv, stream the PGN, and write shards; returns the shard paths.
    Separate from main() for the same reason as selfplay.run() (main() must
    return None for the console-script wrapper)."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    accept = accept_headers_factory(args.min_elo)
    shard_paths: list[Path] = []
    pending: list[Row] = []
    games_in_shard = 0
    stats = {"read": 0, "kept": 0, "skipped_headers": 0, "too_short": 0, "errored": 0}

    def flush() -> None:
        nonlocal games_in_shard
        if not pending:
            return
        path = args.out / f"game-pgn-{len(shard_paths):06d}.npz"
        save_game_shard(
            path,
            list(pending),
            meta={
                "source": args.pgn.name,
                "min_elo": args.min_elo,
                "min_plies": args.min_plies,
                "games": games_in_shard,
                "supervised": True,
                "engine_version": _mcts.version(),
            },
        )
        shard_paths.append(path)
        pending.clear()
        games_in_shard = 0

    with open_pgn(args.pgn) as handle, tqdm(desc="pgn-import", unit="game") as bar:
        while args.max_games == 0 or stats["read"] < args.max_games:
            game = chess.pgn.read_game(handle, Visitor=lambda: _GameExtractor(accept))
            if game is None:
                break  # EOF
            stats["read"] += 1
            bar.update(1)
            if game.skipped:
                stats["skipped_headers"] += 1
            elif game.errored:
                stats["errored"] += 1
            elif len(game.samples) < args.min_plies:
                stats["too_short"] += 1
            else:
                pending.extend(game_to_rows(game.headers, game.samples))
                games_in_shard += 1
                stats["kept"] += 1
                if games_in_shard >= args.games_per_shard:
                    flush()
            if stats["read"] % 10000 == 0:
                bar.set_postfix(kept=stats["kept"], shards=len(shard_paths))
    flush()

    logger.info(
        "read %d games -> kept %d (%d shards); skipped: %d headers, %d too-short, %d parse-errors",
        stats["read"], stats["kept"], len(shard_paths),
        stats["skipped_headers"], stats["too_short"], stats["errored"],
    )
    print(f"{stats['kept']} games -> {len(shard_paths)} shards in {args.out}")
    return shard_paths


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Lichess PGN (.pgn or .pgn.zst) into supervised training shards"
    )
    parser.add_argument("--pgn", required=True, type=Path, help="input .pgn or .pgn.zst")
    parser.add_argument(
        "--out", type=Path, default=Path("data_pretrain/shards"),
        help="output directory for game-pgn-*.npz shards",
    )
    parser.add_argument(
        "--min-elo", type=int, default=1500,
        help="drop games where either player is rated below this (0 = keep all, incl. unrated)",
    )
    parser.add_argument(
        "--min-plies", type=int, default=10,
        help="drop games shorter than this many half-moves (aborts, instant resigns)",
    )
    parser.add_argument(
        "--games-per-shard", type=int, default=2000,
        help="games bundled per .npz shard (larger = fewer, bigger files)",
    )
    parser.add_argument(
        "--max-games", type=int, default=0,
        help="stop after reading this many games (0 = whole file); handy for a quick subset",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()

"""Read-only viewer for self-play `.npz` shards (DESIGN-M6.md section 7.2),
for the web UI's "Self-play" tab.

A shard stores positions and sparse policy targets, not the played-move
sequence itself, so this reconstructs it from the root rows: consecutive
root FENs differ by exactly one ply, and the move connecting them is found
by trying each legal move from the first until one reaches the second
(ignoring move-clock counters). The per-position policy (move -> visit
probability) is similarly decoded from the shard's move indices via
`_mcts.move_indices` — the same reverse lookup a training/analysis script
would need, done once here for display.
"""

from __future__ import annotations

import chess

from chessengine import _mcts
from chessengine.training.dataset import Shard, load_shard


def _find_played_move(fen: str, next_fen: str) -> str | None:
    """The legal move from `fen` that reaches `next_fen`, or None if no
    single legal move does (shouldn't happen for a real shard)."""
    board = chess.Board(fen)
    target = chess.Board(next_fen)
    for move in board.legal_moves:
        trial = board.copy()
        trial.push(move)
        if (
            trial.board_fen() == target.board_fen()
            and trial.turn == target.turn
            and trial.castling_rights == target.castling_rights
            and trial.ep_square == target.ep_square
        ):
            return move.uci()
    return None


def _decode_policy(fen: str, policy_index, policy_prob) -> list[dict]:
    """[{move, prob}, ...] sorted by prob descending: policy_index entries
    are indices into the fixed 4672-slot move space (eval::move_index), not
    move strings, so they're matched back to this position's legal moves."""
    legal = _mcts.legal_moves(fen)
    if not legal:
        return []
    by_index = dict(zip(_mcts.move_indices(fen, legal), legal))
    decoded = [
        {"move": by_index[int(idx)], "prob": float(prob)}
        for idx, prob in zip(policy_index, policy_prob)
        if int(idx) in by_index
    ]
    decoded.sort(key=lambda row: row["prob"], reverse=True)
    return decoded


def game_from_shard(shard: Shard) -> dict:
    """The root-row trajectory as an ordered list of positions: each with
    the move played from it (None for the final, undecided position), its
    SAN, and the recorded search stats + decoded policy."""
    root_indices = [i for i in range(len(shard)) if bool(shard.is_root[i])]
    positions = []
    for k, i in enumerate(root_indices):
        fen = str(shard.fens[i])
        start, end = int(shard.row_offsets[i]), int(shard.row_offsets[i + 1])
        policy = _decode_policy(fen, shard.policy_index[start:end], shard.policy_prob[start:end])

        move = san = None
        if k + 1 < len(root_indices):
            move = _find_played_move(fen, str(shard.fens[root_indices[k + 1]]))
            if move is not None:
                san = chess.Board(fen).san(chess.Move.from_uci(move))

        positions.append(
            {
                "fen": fen,
                "move": move,
                "san": san,
                "search_value": float(shard.search_value[i]),
                "visit_count": int(shard.visit_count[i]),
                "outcome": float(shard.outcome[i]),
                "policy": policy,
            }
        )
    return {"meta": shard.meta, "positions": positions}


def load_game(path: str) -> dict:
    return game_from_shard(load_shard(path))

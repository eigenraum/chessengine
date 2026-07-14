"""Position/move encoding cross-checks (DESIGN-M6.md section 3).

An independent pure-Python reference (python-chess + the documented table
conventions of eval::move_index) is checked against `_mcts.encode_planes`
and `_mcts.move_indices` over a diverse FEN corpus — same philosophy as the
perft gate in test_perft.py.
"""

import chess
import numpy as np
import pytest

from chessengine import _mcts

# Same corpus as test_perft.py's perft gate (Chess Programming Wiki perft
# positions), plus positions exercising ep rights, partial castling rights,
# a high halfmove clock, and black to move.
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION_6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

CORPUS = [
    chess.STARTING_FEN,
    KIWIPETE,
    POSITION_3,
    POSITION_4,
    POSITION_5,
    POSITION_6,
    "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # ep rights (white)
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 4",  # ep rights, black to move next
    "r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1",  # partial castling rights (white kingside only)
    "r3k2r/8/8/8/8/8/8/R3K2R b kq - 0 1",  # partial castling rights, black to move
    "8/8/8/8/8/8/8/4K2k w - - 99 50",  # high halfmove clock
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # black to move
]

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_KNIGHT_OFFSETS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _canon_square(square: int, turn: bool) -> int:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def reference_planes(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    us, them = board.turn, not board.turn
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    for pt_idx, pt in enumerate(_PIECE_ORDER):
        for sq in board.pieces(pt, us):
            csq = _canon_square(sq, us)
            planes[pt_idx, csq // 8, csq % 8] = 1.0
        for sq in board.pieces(pt, them):
            csq = _canon_square(sq, us)
            planes[6 + pt_idx, csq // 8, csq % 8] = 1.0

    if board.has_kingside_castling_rights(us):
        planes[12] = 1.0
    if board.has_queenside_castling_rights(us):
        planes[13] = 1.0
    if board.has_kingside_castling_rights(them):
        planes[14] = 1.0
    if board.has_queenside_castling_rights(them):
        planes[15] = 1.0

    if board.ep_square is not None:
        csq = _canon_square(board.ep_square, us)
        planes[16, csq // 8, csq % 8] = 1.0

    planes[17] = min(1.0, board.halfmove_clock / 100.0)
    planes[18] = 1.0
    return planes


def reference_move_index(fen: str, uci: str) -> int:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    us = board.turn
    from_sq = _canon_square(move.from_square, us)
    to_sq = _canon_square(move.to_square, us)
    dr = (to_sq // 8) - (from_sq // 8)
    df = (to_sq % 8) - (from_sq % 8)

    if move.promotion in _PROMO_PIECES:
        piece_idx = _PROMO_PIECES.index(move.promotion)
        return (64 + (df + 1) * 3 + piece_idx) * 64 + from_sq

    if (abs(dr), abs(df)) in ((1, 2), (2, 1)):
        move_type = 56 + _KNIGHT_OFFSETS.index((dr, df))
        return move_type * 64 + from_sq

    dist = max(abs(dr), abs(df))
    direction = _QUEEN_DIRS.index((_sign(dr), _sign(df)))
    return (direction * 7 + (dist - 1)) * 64 + from_sq


@pytest.mark.parametrize("fen", CORPUS)
def test_planes_match_reference(fen):
    got = _mcts.encode_planes(fen)
    np.testing.assert_array_equal(got, reference_planes(fen))


@pytest.mark.parametrize("fen", CORPUS)
def test_move_indices_valid_and_distinct(fen):
    ucis = _mcts.legal_moves(fen)
    indices = _mcts.move_indices(fen, ucis)
    assert all(0 <= i < _mcts.POLICY_SIZE for i in indices)
    assert len(set(indices)) == len(indices)


@pytest.mark.parametrize("fen", CORPUS)
def test_move_indices_match_reference(fen):
    ucis = _mcts.legal_moves(fen)
    indices = _mcts.move_indices(fen, ucis)
    for uci, index in zip(ucis, indices):
        assert index == reference_move_index(fen, uci)


@pytest.mark.parametrize("fen", CORPUS)
def test_mirror_consistency(fen):
    board = chess.Board(fen)
    mirrored_fen = board.mirror().fen()

    np.testing.assert_array_equal(_mcts.encode_planes(fen), _mcts.encode_planes(mirrored_fen))

    for uci in _mcts.legal_moves(fen):
        move = chess.Move.from_uci(uci)
        mirrored_uci = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        ).uci()
        assert _mcts.move_indices(fen, [uci])[0] == _mcts.move_indices(
            mirrored_fen, [mirrored_uci]
        )[0]


# Hand-computed spot checks (verified once against the implementation, then
# pinned — see the derivation in the M6b PR description). Cover both colors,
# a knight move, castling, an en passant capture, and an underpromotion.
SPOT_CHECKS = [
    (chess.STARTING_FEN, "e2e4", 76),
    (chess.STARTING_FEN, "g1f3", 3718),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 964),
    ("rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", 3172),
    ("7k/8/8/8/8/8/4p3/1K6 b - - 0 1", "e2e1n", 4340),
]


@pytest.mark.parametrize("fen,uci,expected", SPOT_CHECKS)
def test_move_index_spot_checks(fen, uci, expected):
    assert _mcts.move_indices(fen, [uci])[0] == expected

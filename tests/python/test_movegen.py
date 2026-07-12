"""Differential validation of the C++ rules against python-chess.

Compares legal move sets and post-move FENs position by position. python-chess
is the reference; any disagreement is a bug in the C++ core.
"""

import random

import chess
import pytest

from chessengine import _mcts

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION_6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

TRICKY_FENS = [
    chess.STARTING_FEN,
    KIWIPETE,
    POSITION_3,
    POSITION_4,
    POSITION_5,
    POSITION_6,
    # en passant capture is illegal: it would expose the king on the 5th rank
    "8/8/8/K1pP3r/8/8/8/4k3 w - c6 0 1",
    # en passant capture gives check
    "8/8/8/2pP4/8/8/8/K3k3 w - c6 0 1",
    # castling rights but path attacked / blocked
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/4r3/8/8/R3K2R w KQkq - 0 1",
    # promotion with capture, both colors
    "rnbq1bnr/ppppkP1p/8/4p3/8/8/PPPP1PPP/RNBQKBNR w KQ - 1 5",
    "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPKPpP/RNBQ1BNR b kq - 1 5",
    # stalemate and checkmate positions (no legal moves)
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "6rk/6pp/8/8/8/8/8/6RK w - - 0 1",
]


def fen_of(board: chess.Board) -> str:
    # en_passant="fen" records the ep square after any double push, matching
    # the C++ board; the default "legal" omits it unless a capture is possible.
    return board.fen(en_passant="fen")


def assert_same_moves(board: chess.Board) -> None:
    ours = sorted(_mcts.legal_moves(fen_of(board)))
    reference = sorted(move.uci() for move in board.legal_moves)
    assert ours == reference, f"movegen mismatch at {fen_of(board)}"


@pytest.mark.parametrize("fen", TRICKY_FENS)
def test_legal_moves_match(fen):
    assert_same_moves(chess.Board(fen))


def test_random_walk_matches_python_chess():
    """Play random games; every position must agree on moves and FENs."""
    rng = random.Random(42)
    for _ in range(20):
        board = chess.Board()
        for _ in range(120):
            assert_same_moves(board)
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            ours = _mcts.apply_move(fen_of(board), move.uci())
            board.push(move)
            assert ours == fen_of(board)


def test_illegal_move_rejected():
    with pytest.raises(ValueError):
        _mcts.apply_move(chess.STARTING_FEN, "e2e5")


def test_bad_fen_rejected():
    with pytest.raises(ValueError):
        _mcts.legal_moves("not a fen")

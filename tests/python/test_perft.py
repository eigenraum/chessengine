"""Perft gate: the C++ move generator must reproduce known node counts.

Positions from the Chess Programming Wiki perft results page; together they
cover castling (incl. through/out of check), en passant (incl. pins),
promotions, and discovered checks.
"""

import chess
import pytest

from chessengine import _mcts

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION_6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

CASES = [
    (chess.STARTING_FEN, 1, 20),
    (chess.STARTING_FEN, 2, 400),
    (chess.STARTING_FEN, 3, 8_902),
    (chess.STARTING_FEN, 4, 197_281),
    (chess.STARTING_FEN, 5, 4_865_609),
    (KIWIPETE, 1, 48),
    (KIWIPETE, 2, 2_039),
    (KIWIPETE, 3, 97_862),
    (KIWIPETE, 4, 4_085_603),
    (POSITION_3, 1, 14),
    (POSITION_3, 2, 191),
    (POSITION_3, 3, 2_812),
    (POSITION_3, 4, 43_238),
    (POSITION_3, 5, 674_624),
    (POSITION_4, 1, 6),
    (POSITION_4, 2, 264),
    (POSITION_4, 3, 9_467),
    (POSITION_4, 4, 422_333),
    (POSITION_5, 1, 44),
    (POSITION_5, 2, 1_486),
    (POSITION_5, 3, 62_379),
    (POSITION_5, 4, 2_103_487),
    (POSITION_6, 1, 46),
    (POSITION_6, 2, 2_079),
    (POSITION_6, 3, 89_890),
    (POSITION_6, 4, 3_894_594),
]

SLOW_CASES = [
    (chess.STARTING_FEN, 6, 119_060_324),
    (KIWIPETE, 5, 193_690_690),
]


@pytest.mark.parametrize("fen,depth,expected", CASES)
def test_perft(fen, depth, expected):
    assert _mcts.perft(fen, depth) == expected


@pytest.mark.slow
@pytest.mark.parametrize("fen,depth,expected", SLOW_CASES)
def test_perft_deep(fen, depth, expected):
    assert _mcts.perft(fen, depth) == expected

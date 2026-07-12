"""M1 plumbing check: the C++ extension builds and is importable."""

from chessengine import _mcts


def test_extension_importable():
    assert _mcts.version() == "0.1.0"

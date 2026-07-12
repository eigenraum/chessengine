from chessengine import Game
from chessengine.ui.cli import render_board, render_status

STARTPOS_RENDER = """\
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟ 7
6 · · · · · · · · 6
5 · · · · · · · · 5
4 · · · · · · · · 4
3 · · · · · · · · 3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h"""


def test_render_startpos():
    assert render_board(Game()) == STARTPOS_RENDER


def test_render_after_move():
    game = Game()
    game.push("e4")
    rendered = render_board(game)
    lines = rendered.splitlines()
    assert lines[5] == "4 · · · · ♙ · · · 4"  # pawn on e4
    assert lines[7] == "2 ♙ ♙ ♙ ♙ · ♙ ♙ ♙ 2"  # e2 now empty


def test_render_status():
    game = Game()
    assert render_status(game) == "White to move"
    game.push("e4")
    assert render_status(game) == "Black to move"
    for move in ["e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]:
        game.push(move)
    assert render_status(game) == "Game over: 1-0 (checkmate)"

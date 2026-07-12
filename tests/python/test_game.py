import chess
import pytest

from chessengine import Game, IllegalMoveError


def test_new_game_is_startpos():
    game = Game()
    assert game.fen() == chess.STARTING_FEN
    assert game.turn == chess.WHITE
    assert len(game.legal_moves()) == 20
    assert not game.is_over()


def test_push_san_and_uci():
    game = Game()
    game.push("e4")
    game.push("e7e5")
    game.push(chess.Move.from_uci("g1f3"))
    assert game.san_history() == ["e4", "e5", "Nf3"]
    assert game.turn == chess.BLACK


@pytest.mark.parametrize("move", ["e5", "e2e5", "Ke2", "garbage", "e9e9"])
def test_illegal_moves_rejected(move):
    game = Game()
    with pytest.raises(IllegalMoveError):
        game.push(move)
    assert game.fen() == chess.STARTING_FEN  # state untouched


def test_illegal_move_object_rejected():
    game = Game()
    with pytest.raises(IllegalMoveError):
        game.push(chess.Move.from_uci("e2e5"))


def test_checkmate_outcome():
    game = Game()
    for move in ["f3", "e5", "g4", "Qh4"]:  # fool's mate
        game.push(move)
    assert game.is_over()
    outcome = game.outcome()
    assert outcome.winner == chess.BLACK
    assert outcome.termination == chess.Termination.CHECKMATE


def test_custom_fen_start():
    fen = "k7/8/K7/8/8/8/8/7R w - - 0 1"
    game = Game(fen)
    assert game.fen() == fen
    game.push("Rh8#")
    assert game.outcome().winner == chess.WHITE


def test_piece_map_startpos():
    pieces = Game().piece_map()
    assert len(pieces) == 32
    assert pieces[chess.E1] == "K"
    assert pieces[chess.D8] == "q"
    assert chess.E4 not in pieces

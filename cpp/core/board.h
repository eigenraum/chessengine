#pragma once

#include <array>
#include <string>

#include "types.h"

namespace core {

// Position representation: bitboards by color and by piece type, plus a
// mailbox for piece lookup by square. Board is a small value type — the
// search copies it to make a move (copy-make, DESIGN.md section 3), so there
// is no undo.
class Board {
public:
    static const char* const START_FEN;

    Board();                                 // standard starting position
    explicit Board(const std::string& fen);  // throws std::invalid_argument

    std::string fen() const;

    Color side_to_move() const { return stm_; }
    Bitboard pieces(Color c) const { return by_color_[c]; }
    Bitboard pieces(Color c, PieceType pt) const { return by_color_[c] & by_type_[pt]; }
    Bitboard occupied() const { return by_color_[WHITE] | by_color_[BLACK]; }
    PieceType piece_type_on(int sq) const { return PieceType(mailbox_[sq]); }
    int ep_square() const { return ep_square_; }  // -1 if none
    bool can_castle(Color c, bool kingside) const;
    int halfmove_clock() const { return halfmove_clock_; }
    int king_square(Color c) const;

    bool is_attacked(int sq, Color by) const;
    bool in_check() const { return is_attacked(king_square(stm_), opponent(stm_)); }

    // True if the positions are the same for repetition purposes: pieces,
    // side to move, castling rights and ep square; move clocks ignored.
    bool same_position(const Board& other) const {
        return by_color_ == other.by_color_ && by_type_ == other.by_type_ &&
               stm_ == other.stm_ && castling_ == other.castling_ &&
               ep_square_ == other.ep_square_;
    }

    // Applies a pseudo-legal move. Legality filtering happens in movegen by
    // copying the board, applying, and checking the king is not left in check.
    void apply(Move m);

private:
    void put_piece(Color c, PieceType pt, int sq);
    void remove_piece(int sq);
    void update_castling_rights(int from, int to);

    std::array<Bitboard, 2> by_color_{};
    std::array<Bitboard, 6> by_type_{};
    std::array<uint8_t, 64> mailbox_{};  // PieceType per square, NO_PIECE_TYPE if empty
    Color stm_ = WHITE;
    uint8_t castling_ = 0;  // CastlingBit flags, see board.cpp
    int8_t ep_square_ = -1;
    uint16_t halfmove_clock_ = 0;
    uint16_t fullmove_ = 1;
};

}  // namespace core

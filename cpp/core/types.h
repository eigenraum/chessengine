#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace core {

using Bitboard = uint64_t;

enum Color : int { WHITE = 0, BLACK = 1 };

constexpr Color opponent(Color c) { return Color(c ^ 1); }

enum PieceType : int { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE };

// Squares are 0..63 with a1=0, b1=1, ..., h8=63 — the same indexing as
// python-chess, which keeps cross-validation trivial.
constexpr int make_square(int file, int rank) { return rank * 8 + file; }
constexpr int file_of(int sq) { return sq & 7; }
constexpr int rank_of(int sq) { return sq >> 3; }
constexpr Bitboard square_bb(int sq) { return Bitboard(1) << sq; }

inline std::string square_name(int sq) {
    return {char('a' + file_of(sq)), char('1' + rank_of(sq))};
}

// A move fits in 16 bits (from | to << 6 | promotion << 12) so it can be
// stored compactly in tree nodes later. Castling is encoded as the king
// moving two squares, en passant as the pawn capturing onto the ep square —
// both as in UCI notation.
class Move {
public:
    constexpr Move() = default;
    constexpr Move(int from, int to, PieceType promotion = NO_PIECE_TYPE)
        : data_(uint16_t(from | to << 6 |
                         (promotion == NO_PIECE_TYPE ? 0 : promotion) << 12)) {}

    constexpr int from() const { return data_ & 63; }
    constexpr int to() const { return data_ >> 6 & 63; }
    constexpr PieceType promotion() const {
        int p = data_ >> 12;
        return p ? PieceType(p) : NO_PIECE_TYPE;
    }
    constexpr bool is_null() const { return data_ == 0; }
    constexpr bool operator==(const Move&) const = default;
    constexpr uint16_t raw() const { return data_; }

    std::string uci() const {
        std::string s = square_name(from()) + square_name(to());
        if (promotion() != NO_PIECE_TYPE) s += "pnbrqk"[promotion()];
        return s;
    }

    static Move from_uci(const std::string& s) {
        if (s.size() != 4 && s.size() != 5)
            throw std::invalid_argument("bad UCI move: " + s);
        auto square = [&s](size_t i) {
            int file = s[i] - 'a', rank = s[i + 1] - '1';
            if (file < 0 || file > 7 || rank < 0 || rank > 7)
                throw std::invalid_argument("bad UCI move: " + s);
            return make_square(file, rank);
        };
        PieceType promotion = NO_PIECE_TYPE;
        if (s.size() == 5) {
            switch (s[4]) {
                case 'n': promotion = KNIGHT; break;
                case 'b': promotion = BISHOP; break;
                case 'r': promotion = ROOK; break;
                case 'q': promotion = QUEEN; break;
                default: throw std::invalid_argument("bad UCI move: " + s);
            }
        }
        return Move(square(0), square(2), promotion);
    }

private:
    uint16_t data_ = 0;
};

}  // namespace core

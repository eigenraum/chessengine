#include "board.h"

#include <bit>
#include <cctype>
#include <cstdlib>
#include <sstream>

#include "attacks.h"

namespace core {

const char* const Board::START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

namespace {

enum CastlingBit : uint8_t {
    WHITE_KINGSIDE = 1,
    WHITE_QUEENSIDE = 2,
    BLACK_KINGSIDE = 4,
    BLACK_QUEENSIDE = 8,
};

constexpr const char* PIECE_CHARS = "pnbrqk";  // indexed by PieceType

PieceType piece_type_from_char(char lower) {
    for (int pt = PAWN; pt <= KING; ++pt)
        if (PIECE_CHARS[pt] == lower) return PieceType(pt);
    return NO_PIECE_TYPE;
}

// a1/e1/h1 and a8/e8/h8: moving or capturing on these squares loses rights.
constexpr int A1 = 0, E1 = 4, H1 = 7, A8 = 56, E8 = 60, H8 = 63;

}  // namespace

Board::Board() : Board(START_FEN) {}

Board::Board(const std::string& fen) {
    mailbox_.fill(NO_PIECE_TYPE);

    std::istringstream in(fen);
    std::string placement, stm, castling, ep, halfmove, fullmove;
    if (!(in >> placement >> stm >> castling >> ep))
        throw std::invalid_argument("bad FEN (needs at least 4 fields): " + fen);

    int file = 0, rank = 7;
    for (char ch : placement) {
        if (ch == '/') {
            file = 0;
            --rank;
        } else if ('1' <= ch && ch <= '8') {
            file += ch - '0';
        } else {
            PieceType pt = piece_type_from_char(char(std::tolower(ch)));
            if (pt == NO_PIECE_TYPE || file > 7 || rank < 0)
                throw std::invalid_argument("bad FEN placement: " + fen);
            put_piece(std::isupper(ch) ? WHITE : BLACK, pt, make_square(file, rank));
            ++file;
        }
    }

    if (stm == "w") stm_ = WHITE;
    else if (stm == "b") stm_ = BLACK;
    else throw std::invalid_argument("bad FEN side to move: " + fen);

    if (castling != "-") {
        for (char ch : castling) {
            switch (ch) {
                case 'K': castling_ |= WHITE_KINGSIDE; break;
                case 'Q': castling_ |= WHITE_QUEENSIDE; break;
                case 'k': castling_ |= BLACK_KINGSIDE; break;
                case 'q': castling_ |= BLACK_QUEENSIDE; break;
                default: throw std::invalid_argument("bad FEN castling: " + fen);
            }
        }
    }

    if (ep != "-") {
        if (ep.size() != 2 || ep[0] < 'a' || ep[0] > 'h' || ep[1] < '1' || ep[1] > '8')
            throw std::invalid_argument("bad FEN en passant square: " + fen);
        ep_square_ = int8_t(make_square(ep[0] - 'a', ep[1] - '1'));
    }

    if (in >> halfmove) halfmove_clock_ = uint16_t(std::stoi(halfmove));
    if (in >> fullmove) fullmove_ = uint16_t(std::stoi(fullmove));
}

std::string Board::fen() const {
    std::string out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = make_square(file, rank);
            PieceType pt = piece_type_on(sq);
            if (pt == NO_PIECE_TYPE) {
                ++empty;
                continue;
            }
            if (empty) out += char('0' + empty);
            empty = 0;
            char ch = PIECE_CHARS[pt];
            out += (by_color_[WHITE] & square_bb(sq)) ? char(std::toupper(ch)) : ch;
        }
        if (empty) out += char('0' + empty);
        if (rank) out += '/';
    }

    out += stm_ == WHITE ? " w " : " b ";
    if (castling_ & WHITE_KINGSIDE) out += 'K';
    if (castling_ & WHITE_QUEENSIDE) out += 'Q';
    if (castling_ & BLACK_KINGSIDE) out += 'k';
    if (castling_ & BLACK_QUEENSIDE) out += 'q';
    if (!castling_) out += '-';
    out += ' ';
    out += ep_square_ >= 0 ? square_name(ep_square_) : "-";
    out += ' ' + std::to_string(halfmove_clock_) + ' ' + std::to_string(fullmove_);
    return out;
}

bool Board::can_castle(Color c, bool kingside) const {
    uint8_t bit = c == WHITE ? (kingside ? WHITE_KINGSIDE : WHITE_QUEENSIDE)
                             : (kingside ? BLACK_KINGSIDE : BLACK_QUEENSIDE);
    return castling_ & bit;
}

int Board::king_square(Color c) const {
    return std::countr_zero(pieces(c, KING));
}

bool Board::is_attacked(int sq, Color by) const {
    // A `by` pawn attacks sq iff a pawn of the *other* color on sq would
    // attack the pawn's square (pawn attacks are mirror-symmetric).
    if (PAWN_ATTACKS[opponent(by)][sq] & pieces(by, PAWN)) return true;
    if (KNIGHT_ATTACKS[sq] & pieces(by, KNIGHT)) return true;
    if (KING_ATTACKS[sq] & pieces(by, KING)) return true;
    Bitboard occ = occupied();
    if (bishop_attacks(sq, occ) & (pieces(by, BISHOP) | pieces(by, QUEEN))) return true;
    if (rook_attacks(sq, occ) & (pieces(by, ROOK) | pieces(by, QUEEN))) return true;
    return false;
}

void Board::apply(Move m) {
    const int from = m.from(), to = m.to();
    const Color us = stm_;
    const PieceType pt = piece_type_on(from);
    const bool is_capture = piece_type_on(to) != NO_PIECE_TYPE;
    const bool is_en_passant = pt == PAWN && to == ep_square_;

    halfmove_clock_ = (pt == PAWN || is_capture) ? 0 : halfmove_clock_ + 1;

    if (is_en_passant) remove_piece(to + (us == WHITE ? -8 : 8));
    if (is_capture) remove_piece(to);
    remove_piece(from);
    put_piece(us, m.promotion() == NO_PIECE_TYPE ? pt : m.promotion(), to);

    if (pt == KING && to - from == 2) {  // O-O: rook jumps from h to f
        remove_piece(to + 1);
        put_piece(us, ROOK, to - 1);
    } else if (pt == KING && from - to == 2) {  // O-O-O: rook jumps from a to d
        remove_piece(to - 2);
        put_piece(us, ROOK, to + 1);
    }

    update_castling_rights(from, to);
    ep_square_ = int8_t(pt == PAWN && std::abs(to - from) == 16 ? (from + to) / 2 : -1);
    if (us == BLACK) ++fullmove_;
    stm_ = opponent(us);
}

void Board::put_piece(Color c, PieceType pt, int sq) {
    by_color_[c] |= square_bb(sq);
    by_type_[pt] |= square_bb(sq);
    mailbox_[sq] = uint8_t(pt);
}

void Board::remove_piece(int sq) {
    Bitboard bb = square_bb(sq);
    by_color_[WHITE] &= ~bb;
    by_color_[BLACK] &= ~bb;
    by_type_[mailbox_[sq]] &= ~bb;
    mailbox_[sq] = NO_PIECE_TYPE;
}

void Board::update_castling_rights(int from, int to) {
    for (int sq : {from, to}) {
        switch (sq) {
            case E1: castling_ &= ~(WHITE_KINGSIDE | WHITE_QUEENSIDE); break;
            case H1: castling_ &= ~WHITE_KINGSIDE; break;
            case A1: castling_ &= ~WHITE_QUEENSIDE; break;
            case E8: castling_ &= ~(BLACK_KINGSIDE | BLACK_QUEENSIDE); break;
            case H8: castling_ &= ~BLACK_KINGSIDE; break;
            case A8: castling_ &= ~BLACK_QUEENSIDE; break;
        }
    }
}

}  // namespace core

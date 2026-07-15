#include "encode.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <stdexcept>

namespace eval {

namespace {

// Every position is encoded from the side to move's point of view: ranks
// mirror for black (a1<->a8, files unchanged), so the net never needs to
// learn color symmetry (DESIGN-M6.md section 3.1).
int canon(int sq, core::Color stm) { return stm == core::WHITE ? sq : sq ^ 56; }

void fill_plane(std::span<float> out, int plane, float value) {
    std::fill(out.begin() + plane * 64, out.begin() + (plane + 1) * 64, value);
}

// Fixed move-type tables (DESIGN-M6.md section 3.3). The exact orders are a
// free choice: the net learns whatever mapping this file defines, and this
// file is the only encoder, so nothing outside it needs to agree.
constexpr int KNIGHT_DR[8] = {1, 2, 2, 1, -1, -2, -2, -1};
constexpr int KNIGHT_DF[8] = {2, 1, -1, -2, -2, -1, 1, 2};
constexpr int QUEEN_DR[8] = {1, 1, 0, -1, -1, -1, 0, 1};   // N, NE, E, SE, S, SW, W, NW
constexpr int QUEEN_DF[8] = {0, 1, 1, 1, 0, -1, -1, -1};

int sign(int x) { return (x > 0) - (x < 0); }

}  // namespace

void encode_planes(const core::Board& board, std::span<float> out) {
    std::fill(out.begin(), out.end(), 0.0f);

    const core::Color us = board.side_to_move();
    const core::Color them = core::opponent(us);

    for (int pt = core::PAWN; pt <= core::KING; ++pt) {
        core::Bitboard bb = board.pieces(us, core::PieceType(pt));
        while (bb) {
            int sq = std::countr_zero(bb);
            bb &= bb - 1;
            out[pt * 64 + canon(sq, us)] = 1.0f;
        }
        bb = board.pieces(them, core::PieceType(pt));
        while (bb) {
            int sq = std::countr_zero(bb);
            bb &= bb - 1;
            out[(6 + pt) * 64 + canon(sq, us)] = 1.0f;
        }
    }

    if (board.can_castle(us, true)) fill_plane(out, 12, 1.0f);
    if (board.can_castle(us, false)) fill_plane(out, 13, 1.0f);
    if (board.can_castle(them, true)) fill_plane(out, 14, 1.0f);
    if (board.can_castle(them, false)) fill_plane(out, 15, 1.0f);

    if (board.ep_square() >= 0) out[16 * 64 + canon(board.ep_square(), us)] = 1.0f;

    fill_plane(out, 17, std::min(1.0f, float(board.halfmove_clock()) / 100.0f));
    fill_plane(out, 18, 1.0f);
}

int move_index(const core::Board& board, core::Move move) {
    const core::Color stm = board.side_to_move();
    const int from = canon(move.from(), stm);
    const int to = canon(move.to(), stm);
    const int dr = core::rank_of(to) - core::rank_of(from);
    const int df = core::file_of(to) - core::file_of(from);

    // Underpromotion: queen promotions fall through to the queen-move case
    // below (AlphaZero convention — see DESIGN-M6.md section 3.3).
    const core::PieceType promo = move.promotion();
    if (promo == core::KNIGHT || promo == core::BISHOP || promo == core::ROOK) {
        const int piece_idx = promo == core::KNIGHT ? 0 : promo == core::BISHOP ? 1 : 2;
        return (64 + (df + 1) * 3 + piece_idx) * 64 + from;
    }

    if ((std::abs(dr) == 1 && std::abs(df) == 2) || (std::abs(dr) == 2 && std::abs(df) == 1)) {
        for (int i = 0; i < 8; ++i)
            if (KNIGHT_DR[i] == dr && KNIGHT_DF[i] == df) return (56 + i) * 64 + from;
    }

    // Queen move: covers ordinary sliding moves, queen promotions, castling
    // (the king's two-square move), and en passant (an ordinary diagonal
    // pawn move).
    const int dist = std::max(std::abs(dr), std::abs(df));
    const int sdr = sign(dr), sdf = sign(df);
    for (int dir = 0; dir < 8; ++dir)
        if (QUEEN_DR[dir] == sdr && QUEEN_DF[dir] == sdf) return (dir * 7 + (dist - 1)) * 64 + from;

    throw std::logic_error("move_index: no case matched (should be unreachable for legal moves)");
}

}  // namespace eval

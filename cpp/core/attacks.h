#pragma once

#include <array>
#include <utility>

#include "types.h"

// Attack sets for every piece type. Leaper attacks (knight, king, pawn) come
// from compile-time tables; slider attacks (bishop, rook, queen) walk their
// rays until they hit a blocker. The classical blocker loop is the simple
// choice over magic bitboards (DESIGN.md section 3) — it can be swapped out
// behind these same functions if profiling ever demands it.

namespace core {

namespace detail {

using Step = std::pair<int, int>;  // (file delta, rank delta)

constexpr Bitboard step_target(int sq, Step step) {
    int file = file_of(sq) + step.first, rank = rank_of(sq) + step.second;
    bool on_board = 0 <= file && file < 8 && 0 <= rank && rank < 8;
    return on_board ? square_bb(make_square(file, rank)) : 0;
}

template <size_t N>
constexpr std::array<Bitboard, 64> leaper_table(const std::array<Step, N>& steps) {
    std::array<Bitboard, 64> table{};
    for (int sq = 0; sq < 64; ++sq)
        for (Step step : steps) table[sq] |= step_target(sq, step);
    return table;
}

constexpr Bitboard ray_attacks(int sq, Bitboard occupied, Step dir) {
    Bitboard attacks = 0;
    int file = file_of(sq) + dir.first, rank = rank_of(sq) + dir.second;
    while (0 <= file && file < 8 && 0 <= rank && rank < 8) {
        int target = make_square(file, rank);
        attacks |= square_bb(target);
        if (occupied & square_bb(target)) break;  // blocker: ray stops here
        file += dir.first;
        rank += dir.second;
    }
    return attacks;
}

}  // namespace detail

inline constexpr auto KNIGHT_ATTACKS = detail::leaper_table<8>(
    {{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}});

inline constexpr auto KING_ATTACKS = detail::leaper_table<8>(
    {{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}});

// PAWN_ATTACKS[color][sq]: squares attacked by a `color` pawn standing on sq.
inline constexpr std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS = {
    detail::leaper_table<2>({{{-1, 1}, {1, 1}}}),    // WHITE
    detail::leaper_table<2>({{{-1, -1}, {1, -1}}}),  // BLACK
};

constexpr Bitboard bishop_attacks(int sq, Bitboard occupied) {
    return detail::ray_attacks(sq, occupied, {1, 1}) |
           detail::ray_attacks(sq, occupied, {1, -1}) |
           detail::ray_attacks(sq, occupied, {-1, 1}) |
           detail::ray_attacks(sq, occupied, {-1, -1});
}

constexpr Bitboard rook_attacks(int sq, Bitboard occupied) {
    return detail::ray_attacks(sq, occupied, {1, 0}) |
           detail::ray_attacks(sq, occupied, {-1, 0}) |
           detail::ray_attacks(sq, occupied, {0, 1}) |
           detail::ray_attacks(sq, occupied, {0, -1});
}

constexpr Bitboard queen_attacks(int sq, Bitboard occupied) {
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied);
}

}  // namespace core

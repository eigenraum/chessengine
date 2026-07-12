#include "material.h"

#include <bit>

namespace eval {

namespace {
// PAWN, KNIGHT, BISHOP, ROOK, QUEEN
constexpr int PIECE_VALUES[] = {100, 320, 330, 500, 900};
}

void MaterialEvaluator::evaluate(std::span<const core::Board* const> positions,
                                 std::span<float> values_out) {
    for (size_t i = 0; i < positions.size(); ++i) {
        const core::Board& board = *positions[i];
        int cp = 0;
        for (int pt = core::PAWN; pt <= core::QUEEN; ++pt) {
            cp += PIECE_VALUES[pt] *
                  (std::popcount(board.pieces(core::WHITE, core::PieceType(pt))) -
                   std::popcount(board.pieces(core::BLACK, core::PieceType(pt))));
        }
        if (board.side_to_move() == core::BLACK) cp = -cp;
        values_out[i] = centipawns_to_win_prob(float(cp));
    }
}

}  // namespace eval

#include "material.h"

#include <algorithm>
#include <bit>

namespace eval {

namespace {
// PAWN, KNIGHT, BISHOP, ROOK, QUEEN
constexpr int PIECE_VALUES[] = {100, 320, 330, 500, 900};
}

void MaterialEvaluator::evaluate(std::span<const EvalRequest> batch) {
    for (const EvalRequest& request : batch) {
        const core::Board& board = *request.board;
        int cp = 0;
        for (int pt = core::PAWN; pt <= core::QUEEN; ++pt) {
            cp += PIECE_VALUES[pt] *
                  (std::popcount(board.pieces(core::WHITE, core::PieceType(pt))) -
                   std::popcount(board.pieces(core::BLACK, core::PieceType(pt))));
        }
        if (board.side_to_move() == core::BLACK) cp = -cp;
        *request.value_out = centipawns_to_win_prob(float(cp));

        if (!request.moves.empty()) {
            const float prior = 1.0f / float(request.moves.size());
            std::fill(request.priors_out.begin(), request.priors_out.end(), prior);
        }
    }
}

}  // namespace eval

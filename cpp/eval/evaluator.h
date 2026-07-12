#pragma once

#include <algorithm>
#include <cmath>
#include <span>

#include "core/board.h"

namespace eval {

// Batch evaluation interface (DESIGN.md section 4.3). Implementations score
// whole batches at once; values are win probabilities in [0, 1] from the
// perspective of the side to move. The material heuristic implements this
// now; a PyTorch policy/value net implements the same interface later.
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void evaluate(std::span<const core::Board* const> positions,
                          std::span<float> values_out) = 0;
};

// Standard Elo-style logistic mapping between centipawns and win probability:
// +400 cp ~ 90% win chance. Used for display and for squashing heuristic
// scores; the one constant every cp<->probability conversion goes through.
inline constexpr float CP_SCALE = 400.0f;

inline float centipawns_to_win_prob(float cp) {
    return 1.0f / (1.0f + std::pow(10.0f, -cp / CP_SCALE));
}

inline float win_prob_to_centipawns(float p) {
    p = std::clamp(p, 0.001f, 0.999f);
    return -CP_SCALE * std::log10(1.0f / p - 1.0f);
}

}  // namespace eval

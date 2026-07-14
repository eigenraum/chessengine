#pragma once

#include "evaluator.h"

namespace eval {

// Cheap material-count heuristic: piece-value balance in centipawns, squashed
// to a win probability. Deliberately primitive — it exists to make the search
// plumbing real until a learned evaluator replaces it (DESIGN.md section 4.3).
class MaterialEvaluator : public Evaluator {
public:
    void evaluate(std::span<const EvalRequest> batch) override;
};

}  // namespace eval

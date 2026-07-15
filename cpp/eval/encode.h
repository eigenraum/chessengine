#pragma once

#include <span>

#include "core/board.h"

namespace eval {

inline constexpr int PLANES = 19;
inline constexpr int POLICY_SIZE = 73 * 64;  // 4672

// Writes PLANES*64 floats (plane-major, then square 0..63 in canonical
// orientation) for `board` into `out` (DESIGN-M6.md section 3.2).
// out.size() must be PLANES*64.
void encode_planes(const core::Board& board, std::span<float> out);

// Index of `move` in the flat [0, POLICY_SIZE) policy output, for `board`'s
// side to move (DESIGN-M6.md section 3.3).
int move_index(const core::Board& board, core::Move move);

}  // namespace eval

#pragma once

#include <array>
#include <cstdint>

#include "board.h"

namespace core {

// Fixed-capacity move container: no heap allocation in the search hot path.
// 256 comfortably exceeds the maximum number of legal moves in any position.
class MoveList {
public:
    void push(Move m) { moves_[size_++] = m; }
    const Move* begin() const { return moves_.data(); }
    const Move* end() const { return moves_.data() + size_; }
    int size() const { return size_; }

private:
    std::array<Move, 256> moves_;
    int size_ = 0;
};

// All legal moves in the position: pseudo-legal generation followed by
// make-and-check filtering. Simple and clearly correct; fast enough
// (DESIGN.md section 3).
MoveList generate_legal(const Board& board);

// Number of positions reachable in exactly `depth` plies. Used to
// cross-validate the move generator against python-chess.
uint64_t perft(const Board& board, int depth);

}  // namespace core

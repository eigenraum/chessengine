#pragma once

#include <atomic>
#include <cstdint>

#include "core/types.h"

namespace mcts {

enum class ExpandState : uint8_t { UNEXPANDED, EXPANDING, EXPANDED };

// One search-tree node, 32 bytes (two per cache line). Children are
// contiguous in the arena: indices first_child .. first_child+num_children-1.
//
// Statistics are atomics so worker threads update them without locks.
// `value_sum` accumulates outcomes from the perspective of the player who
// moved INTO this node — that way a parent picks the child with the highest
// Q = value_sum/visits directly.
//
// `expand_state` is a tiny per-node spinlock for expansion: the thread that
// wins the UNEXPANDED -> EXPANDING race generates the moves, publishes
// first_child/num_children, then stores EXPANDED (release). Losers simply
// treat the node as a leaf for their simulation.
struct Node {
    std::atomic<uint32_t> visits{0};
    std::atomic<int32_t> virtual_loss{0};
    std::atomic<double> value_sum{0.0};
    float prior{0.0f};        // P(move | parent); uniform until a policy net lands
    uint32_t first_child{0};
    uint16_t num_children{0};
    core::Move move{};        // the move that led here (from the parent)
    std::atomic<ExpandState> expand_state{ExpandState::UNEXPANDED};
};

}  // namespace mcts

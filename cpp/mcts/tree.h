#pragma once

#include <atomic>
#include <memory>

#include "core/board.h"
#include "node.h"

namespace mcts {

// Node arena. Fixed capacity, allocated up front: a block of children is
// reserved with a single atomic bump, nodes never move (so other threads can
// keep reading while allocations happen), and memory use is bounded. If the
// arena fills up the search keeps running — nodes just stop expanding and
// keep being evaluated as leaves; size max_nodes generously.
class Tree {
public:
    static constexpr uint32_t NO_NODE = UINT32_MAX;

    Tree(const core::Board& root_board, uint32_t max_nodes)
        : root_board_(root_board),
          nodes_(std::make_unique<Node[]>(max_nodes)),
          capacity_(max_nodes),
          size_(1) {}  // node 0 is the root

    Node& operator[](uint32_t index) { return nodes_[index]; }
    const Node& operator[](uint32_t index) const { return nodes_[index]; }

    uint32_t root() const { return 0; }
    const core::Board& root_board() const { return root_board_; }
    uint32_t size() const { return size_.load(std::memory_order_relaxed); }
    uint32_t capacity() const { return capacity_; }

    // Reserves a contiguous block of `count` fresh nodes and returns the
    // index of the first one, or NO_NODE if the arena is full.
    uint32_t allocate(uint32_t count) {
        uint32_t first = size_.fetch_add(count, std::memory_order_relaxed);
        if (first + count > capacity_) {
            size_.store(capacity_, std::memory_order_relaxed);  // clamp; stays full
            return NO_NODE;
        }
        return first;
    }

private:
    core::Board root_board_;
    std::unique_ptr<Node[]> nodes_;
    uint32_t capacity_;
    std::atomic<uint32_t> size_;
};

// Tree reuse (DESIGN.md section 4.5): builds a fresh tree rooted at the old
// root's child reached by `move`, copying that whole subtree (statistics
// included, virtual loss zeroed) into a new arena. Copying compacts memory
// and frees everything outside the played line; it runs once per game move,
// off the hot path. `new_root_board` must be the old root board with `move`
// applied. If the child was never visited, the result is simply a fresh
// single-node tree. Must not be called while a search is running.
std::unique_ptr<Tree> extract_subtree(const Tree& old_tree, core::Move move,
                                      const core::Board& new_root_board,
                                      uint32_t max_nodes);

}  // namespace mcts

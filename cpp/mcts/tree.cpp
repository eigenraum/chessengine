#include "tree.h"

#include <deque>
#include <utility>

namespace mcts {

namespace {

// Copies one node's statistics. The structural fields (first_child,
// num_children, expand_state) are set by the BFS below; virtual loss is
// zero on an idle tree by construction.
void copy_stats(const Node& src, Node& dst) {
    dst.visits.store(src.visits.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    dst.value_sum.store(src.value_sum.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    dst.prior = src.prior;
    dst.move = src.move;
}

}  // namespace

std::unique_ptr<Tree> extract_subtree(const Tree& old_tree, core::Move move,
                                      const core::Board& new_root_board,
                                      uint32_t max_nodes) {
    auto fresh = std::make_unique<Tree>(new_root_board, max_nodes);

    // Find the old root's child for `move`; without one (or without visits)
    // there is nothing worth carrying over.
    const Node& old_root = old_tree[old_tree.root()];
    if (old_root.expand_state.load(std::memory_order_relaxed) != ExpandState::EXPANDED)
        return fresh;
    uint32_t old_child = Tree::NO_NODE;
    for (uint32_t i = 0; i < old_root.num_children; ++i) {
        if (old_tree[old_root.first_child + i].move == move) {
            old_child = old_root.first_child + i;
            break;
        }
    }
    if (old_child == Tree::NO_NODE ||
        old_tree[old_child].visits.load(std::memory_order_relaxed) == 0)
        return fresh;

    // Breadth-first copy, allocating each node's child block on arrival —
    // this preserves the children-are-contiguous invariant. The subtree is
    // at most as large as the old arena, so allocation cannot fail.
    Tree& tree = *fresh;
    copy_stats(old_tree[old_child], tree[tree.root()]);
    std::deque<std::pair<uint32_t, uint32_t>> pending{{old_child, tree.root()}};
    while (!pending.empty()) {
        auto [old_index, new_index] = pending.front();
        pending.pop_front();
        const Node& src = old_tree[old_index];
        Node& dst = tree[new_index];

        if (src.expand_state.load(std::memory_order_relaxed) != ExpandState::EXPANDED) {
            continue;  // leaf; dst stays UNEXPANDED
        }
        if (src.num_children == 0) {  // terminal (mate/stalemate)
            dst.expand_state.store(ExpandState::EXPANDED, std::memory_order_relaxed);
            continue;
        }

        uint32_t first = tree.allocate(src.num_children);
        dst.first_child = first;
        dst.num_children = src.num_children;
        dst.expand_state.store(ExpandState::EXPANDED, std::memory_order_relaxed);
        for (uint32_t i = 0; i < src.num_children; ++i) {
            copy_stats(old_tree[src.first_child + i], tree[first + i]);
            pending.emplace_back(src.first_child + i, first + i);
        }
    }
    return fresh;
}

}  // namespace mcts

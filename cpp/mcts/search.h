#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/board.h"
#include "eval/evaluator.h"
#include "eval_queue.h"
#include "tree.h"

namespace mcts {

struct SearchConfig {
    int workers = 1;      // 1 = fully sequential reference mode
    int batch_size = 8;   // max leaves per evaluation batch
    uint32_t max_nodes = 1u << 22;  // arena capacity (~128 MB of nodes)
    uint64_t seed = 0;              // reserved for root noise (M6)
};

struct SearchLimits {
    int max_time_ms = 5000;         // <= 0 = no time limit (infinite analysis)
    int64_t max_simulations = -1;   // -1 = no simulation limit
    // Converged = over the last `convergence_window` simulations the root
    // centipawn evaluation drifted less than the threshold AND the best move
    // did not change. <= 0 disables early stopping.
    int convergence_window = 2000;
    int convergence_cp_threshold = 5;
    // Per-search algorithm parameters (not structural, so they live here and
    // can change between searches without rebuilding the engine).
    float c_puct = 1.5f;  // PUCT exploration constant
    int virtual_loss = 1;
    // Root exploration noise (DESIGN-M6.md section 4.3): self-play wants it,
    // interactive play and analysis don't. 0 = off.
    float root_noise_eps = 0.0f;
    float root_dirichlet_alpha = 0.3f;  // concentration; 0.3 is chess-standard
};

struct SearchStats {
    uint64_t simulations = 0;
    uint64_t nodes = 0;
    float root_value = 0.5f;  // win probability, side-to-move's view
    int root_cp = 0;          // the same, as centipawns
    std::string best_move;    // UCI; empty if the position has no legal moves
    std::vector<std::string> pv;
    int64_t elapsed_ms = 0;
};

struct SearchResult : SearchStats {
    std::string stop_reason;  // "time" | "converged" | "simulations" |
                              // "interrupted" | "no_legal_moves"
};

// Training-data export (DESIGN.md section 4.6): one row per exported node.
// values are win probabilities from the perspective of the side to move in
// fens[i]; moves[i]/child_visits[i] give the root-of-policy visit
// distribution over that node's explored children.
struct TreeSnapshot {
    std::vector<std::string> fens;
    std::vector<uint64_t> visit_counts;
    std::vector<float> values;
    std::vector<std::vector<std::string>> moves;
    std::vector<std::vector<uint32_t>> child_visits;
};

// Live debug view of the search tree (DESIGN-VISU.md section 5.2): the
// `max_nodes` most-visited nodes of the subtree at `root_path`, as flat
// parallel arrays. parent[i] indexes into these same arrays and is always
// < i (-1 for the subtree root, which is row 0). q[i] = value_sum/visits is
// the win frequency from the perspective of the player who moved INTO node i
// (the node-statistics convention). children_total[i] counts all legal
// children, so a client can tell how many were cut off by the node budget.
//
// Unlike snapshot(), this may be called WHILE a search is running: nodes
// never move, children are only traversed once expand_state is EXPANDED
// (acquire, pairing with the release store in maybe_expand), and the
// visit/value loads are racy-but-monotonic like stats() — numbers may be a
// few simulations stale or mutually inconsistent, which is fine for display.
struct TreeView {
    std::vector<int32_t> parent;
    std::vector<std::string> move;  // UCI; "" for the subtree root
    std::vector<uint32_t> visits;
    std::vector<float> q;
    std::vector<float> prior;
    std::vector<uint32_t> children_total;
};

// One MCTS search over one tree, tree-parallel with virtual loss: all worker
// threads share the arena, visit counts and value sums are atomics, and the
// expand CAS plus virtual loss keep them coordinated. A controller thread
// watches the termination conditions. workers = 1 is the fully sequential
// reference: one worker, and virtual loss cancels out exactly.
class Search {
public:
    Search(const SearchConfig& config, eval::Evaluator& evaluator);
    ~Search();

    void set_position(const core::Board& board);  // starts a fresh tree

    // Plays `move` on the internal tree: the matching subtree of the current
    // root is carried over, everything else is dropped. Throws
    // std::invalid_argument if the move is not legal in the root position.
    void advance(core::Move move);

    // Exports the tree for training: nodes with at least min_visits, at most
    // max_depth plies below the root.
    TreeSnapshot snapshot(uint32_t min_visits, int max_depth) const;

    // Live view for the GUI (see TreeView). Best-first by visit count from
    // the node reached by walking root_path (empty = the search root); nodes
    // below min_visits are skipped. Empty view if the path is not in the
    // tree. Safe to call while a search is running.
    TreeView tree_view(uint32_t max_nodes, uint32_t min_visits,
                       const std::vector<core::Move>& root_path) const;

    SearchResult run(const SearchLimits& limits);  // blocking
    void start(const SearchLimits& limits);        // non-blocking, for the GUI
    SearchResult stop();                           // interrupt (if running) + collect
    bool running() const { return running_.load(std::memory_order_acquire); }

    void request_stop() { stop_requested_.store(true, std::memory_order_relaxed); }
    SearchStats stats() const;  // safe to call while a search is running

private:
    SearchResult run_controller(const SearchLimits& limits);
    void worker_loop(const SearchLimits& limits);
    void descend(std::vector<std::vector<uint32_t>>& paths,
                 std::vector<core::Board>& out_boards,
                 std::vector<std::vector<core::Move>>& out_moves);
    uint32_t select_child(const Node& parent) const;
    void maybe_expand(uint32_t index, const core::Board& board);
    void backprop(const std::vector<uint32_t>& path, float leaf_value);
    int64_t elapsed_ms() const;

    SearchConfig config_;
    // The running search's limits. Written by start() before the controller
    // (and thus the workers) exist, so the hot-path reads of c_puct and
    // virtual_loss need no synchronization.
    SearchLimits limits_;
    EvalQueue queue_;
    std::unique_ptr<Tree> tree_;
    std::atomic<bool> stop_requested_{false};  // external: user/GUI interrupt
    std::atomic<bool> stop_workers_{false};    // internal: controller -> workers
    std::atomic<uint64_t> simulations_{0};     // completed simulations
    std::atomic<int64_t> tickets_{0};          // started simulations (max_simulations)
    std::chrono::steady_clock::time_point started_at_;
    uint64_t searches_started_ = 0;  // seeds root Dirichlet noise (fresh per search)

    std::thread controller_;
    std::atomic<bool> running_{false};
    SearchResult result_;  // written by the controller; read after joining it
};

}  // namespace mcts

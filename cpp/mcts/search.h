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
    float c_puct = 1.5f;  // PUCT exploration constant
    int virtual_loss = 1;
    uint32_t max_nodes = 1u << 22;  // arena capacity (~128 MB of nodes)
    uint64_t seed = 0;              // reserved for root noise (M6)
};

struct SearchLimits {
    int max_time_ms = 5000;
    int64_t max_simulations = -1;   // -1 = no simulation limit
    // Converged = over the last `convergence_window` simulations the root
    // centipawn evaluation drifted less than the threshold AND the best move
    // did not change. <= 0 disables early stopping.
    int convergence_window = 2000;
    int convergence_cp_threshold = 5;
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
                 std::vector<core::Board>& out_boards);
    uint32_t select_child(const Node& parent) const;
    void maybe_expand(uint32_t index, const core::Board& board);
    void backprop(const std::vector<uint32_t>& path, float leaf_value);
    int64_t elapsed_ms() const;

    SearchConfig config_;
    EvalQueue queue_;
    std::unique_ptr<Tree> tree_;
    std::atomic<bool> stop_requested_{false};  // external: user/GUI interrupt
    std::atomic<bool> stop_workers_{false};    // internal: controller -> workers
    std::atomic<uint64_t> simulations_{0};     // completed simulations
    std::atomic<int64_t> tickets_{0};          // started simulations (max_simulations)
    std::chrono::steady_clock::time_point started_at_;

    std::thread controller_;
    std::atomic<bool> running_{false};
    SearchResult result_;  // written by the controller; read after joining it
};

}  // namespace mcts

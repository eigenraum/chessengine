#include "search.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <deque>
#include <stdexcept>

#include "core/movegen.h"

namespace mcts {

namespace {

// Search-internal draw detection, conservative subset: bare kings or a single
// minor piece in total. The evaluator scores other dead positions near 0.5
// anyway; the exact FIDE rules live in python-chess at the game level.
bool insufficient_material(const core::Board& board) {
    if (std::popcount(board.occupied()) > 3) return false;
    core::Bitboard heavy = 0;
    for (core::Color c : {core::WHITE, core::BLACK})
        heavy |= board.pieces(c, core::PAWN) | board.pieces(c, core::ROOK) |
                 board.pieces(c, core::QUEEN);
    return heavy == 0;
}

// Twofold repetition within the simulation path approximates threefold
// (DESIGN.md section 3): if the position occurred earlier on this path, score
// it as a draw.
bool repeats_earlier_position(const std::vector<core::Board>& boards) {
    const core::Board& current = boards.back();
    for (size_t i = 0; i + 1 < boards.size(); ++i)
        if (boards[i].same_position(current)) return true;
    return false;
}

bool draw_by_rule(const core::Board& board, const std::vector<core::Board>& boards) {
    return board.halfmove_clock() >= 100 || insufficient_material(board) ||
           repeats_earlier_position(boards);
}

}  // namespace

Search::Search(const SearchConfig& config, eval::Evaluator& evaluator)
    : config_(config), queue_(evaluator, size_t(config.batch_size)) {}

void Search::set_position(const core::Board& board) {
    tree_ = std::make_unique<Tree>(board, config_.max_nodes);
}

void Search::maybe_expand(uint32_t index, const core::Board& board) {
    Node& node = (*tree_)[index];
    ExpandState expected = ExpandState::UNEXPANDED;
    if (!node.expand_state.compare_exchange_strong(expected, ExpandState::EXPANDING,
                                                   std::memory_order_acq_rel))
        return;  // lost the race, or already expanded

    core::MoveList moves = core::generate_legal(board);
    uint32_t first = 0;
    if (moves.size() > 0) {
        first = tree_->allocate(uint32_t(moves.size()));
        if (first == Tree::NO_NODE) {
            // Arena full: stay a leaf (still evaluated), retry never succeeds
            // but the search keeps making progress on the existing tree.
            node.expand_state.store(ExpandState::UNEXPANDED, std::memory_order_release);
            return;
        }
        const float prior = 1.0f / float(moves.size());  // uniform until a policy net
        for (int i = 0; i < moves.size(); ++i) {
            Node& child = (*tree_)[first + uint32_t(i)];
            child.move = moves.begin()[i];
            child.prior = prior;
        }
    }
    node.first_child = first;
    node.num_children = uint16_t(moves.size());
    node.expand_state.store(ExpandState::EXPANDED, std::memory_order_release);
}

uint32_t Search::select_child(const Node& parent) const {
    const Tree& tree = *tree_;
    const double sqrt_parent_visits =
        std::sqrt(double(std::max(parent.visits.load(std::memory_order_relaxed), 1u)));

    double best_score = -1.0;
    uint32_t best_index = parent.first_child;
    for (uint32_t i = 0; i < parent.num_children; ++i) {
        const uint32_t index = parent.first_child + i;
        const Node& child = tree[index];
        // Virtual loss counts as visits that returned losses: it lowers Q and
        // raises N, steering concurrent simulations apart.
        const double n = double(child.visits.load(std::memory_order_relaxed)) +
                         double(std::max(child.virtual_loss.load(std::memory_order_relaxed), 0));
        const double q =
            n > 0 ? child.value_sum.load(std::memory_order_relaxed) / n : 0.5;
        const double score =
            q + config_.c_puct * child.prior * sqrt_parent_visits / (1.0 + n);
        if (score > best_score) {
            best_score = score;
            best_index = index;
        }
    }
    return best_index;
}

// The selection phase of one simulation: walk down with PUCT + virtual loss.
// Terminal and draw leaves are backpropagated immediately; a leaf that needs
// evaluation is appended to paths/boards instead, parked on its virtual loss
// until the caller has a full batch.
void Search::descend(std::vector<std::vector<uint32_t>>& paths,
                     std::vector<core::Board>& out_boards) {
    Tree& tree = *tree_;
    core::Board board = tree.root_board();
    std::vector<uint32_t> path{tree.root()};
    std::vector<core::Board> boards{board};
    tree[tree.root()].virtual_loss.fetch_add(config_.virtual_loss,
                                             std::memory_order_relaxed);

    // Values are from the perspective of the player who moved INTO the leaf
    // (the convention node statistics use).
    float value_for_mover;
    for (;;) {
        Node& node = tree[path.back()];

        if (draw_by_rule(board, boards)) {
            value_for_mover = 0.5f;
            break;
        }

        bool arrived_at_leaf = false;
        if (node.expand_state.load(std::memory_order_acquire) != ExpandState::EXPANDED) {
            maybe_expand(path.back(), board);
            arrived_at_leaf = true;
        }

        const bool expanded =
            node.expand_state.load(std::memory_order_acquire) == ExpandState::EXPANDED;
        if (expanded && node.num_children == 0) {
            // No legal moves: the mover delivered mate, or it is stalemate.
            value_for_mover = board.in_check() ? 1.0f : 0.5f;
            break;
        }
        if (arrived_at_leaf || !expanded) {
            // Fresh leaf (or expansion raced/arena full): needs evaluation.
            paths.push_back(std::move(path));
            out_boards.push_back(board);
            return;
        }

        uint32_t child_index = select_child(node);
        Node& child = tree[child_index];
        child.virtual_loss.fetch_add(config_.virtual_loss, std::memory_order_relaxed);
        board.apply(child.move);
        path.push_back(child_index);
        boards.push_back(board);
    }

    backprop(path, value_for_mover);
    simulations_.fetch_add(1, std::memory_order_relaxed);
}

void Search::backprop(const std::vector<uint32_t>& path, float leaf_value) {
    float value = leaf_value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node& node = (*tree_)[*it];
        node.visits.fetch_add(1, std::memory_order_relaxed);
        node.value_sum.fetch_add(double(value), std::memory_order_relaxed);
        node.virtual_loss.fetch_sub(config_.virtual_loss, std::memory_order_relaxed);
        value = 1.0f - value;  // one ply up, the other player's perspective
    }
}

Search::~Search() {
    request_stop();
    if (controller_.joinable()) controller_.join();
}

SearchResult Search::run(const SearchLimits& limits) {
    start(limits);
    controller_.join();
    return result_;
}

void Search::start(const SearchLimits& limits) {
    if (!tree_) throw std::logic_error("no position set");
    if (running()) throw std::logic_error("search already running");
    if (controller_.joinable()) controller_.join();  // reap a finished search

    stop_requested_.store(false, std::memory_order_relaxed);
    running_.store(true, std::memory_order_release);
    controller_ = std::thread([this, limits] {
        result_ = run_controller(limits);
        running_.store(false, std::memory_order_release);
    });
}

SearchResult Search::stop() {
    request_stop();
    if (controller_.joinable()) controller_.join();
    return result_;
}

// The controller owns the search lifecycle: it expands the root, launches the
// workers, watches the termination conditions, and collects the result.
SearchResult Search::run_controller(const SearchLimits& limits) {
    stop_workers_.store(false, std::memory_order_relaxed);
    simulations_.store(0, std::memory_order_relaxed);
    tickets_.store(0, std::memory_order_relaxed);
    started_at_ = std::chrono::steady_clock::now();

    maybe_expand(tree_->root(), tree_->root_board());
    std::string reason;
    if ((*tree_)[tree_->root()].num_children == 0) {
        reason = "no_legal_moves";
    } else {
        std::vector<std::thread> workers;
        for (int i = 0; i < config_.workers; ++i)
            workers.emplace_back([this, &limits] { worker_loop(limits); });

        // Convergence tracking: snapshot (cp, best move) every window/8
        // simulations; converged once 9 snapshots (= one full window) agree.
        const uint64_t stride = uint64_t(std::max(limits.convergence_window / 8, 1));
        std::deque<std::pair<int, std::string>> snapshots;
        uint64_t last_snapshot = 0;

        for (;;) {
            if (stop_requested_.load(std::memory_order_relaxed)) {
                reason = "interrupted";
                break;
            }
            if (elapsed_ms() >= limits.max_time_ms) {
                reason = "time";
                break;
            }
            const uint64_t sims = simulations_.load(std::memory_order_relaxed);
            if (limits.max_simulations >= 0 && int64_t(sims) >= limits.max_simulations) {
                reason = "simulations";
                break;
            }

            if (limits.convergence_window > 0 && sims - last_snapshot >= stride) {
                last_snapshot = sims;
                SearchStats snapshot = stats();
                snapshots.emplace_back(snapshot.root_cp, snapshot.best_move);
                if (snapshots.size() > 9) snapshots.pop_front();
                if (snapshots.size() == 9) {
                    auto [min_cp, max_cp] = std::minmax_element(
                        snapshots.begin(), snapshots.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
                    const bool value_stalled = max_cp->first - min_cp->first <=
                                               limits.convergence_cp_threshold;
                    const bool move_stable = std::all_of(
                        snapshots.begin(), snapshots.end(), [&](const auto& s) {
                            return s.second == snapshots.front().second;
                        });
                    if (value_stalled && move_stable) {
                        reason = "converged";
                        break;
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        stop_workers_.store(true, std::memory_order_relaxed);
        for (std::thread& worker : workers) worker.join();
    }

    SearchResult result;
    static_cast<SearchStats&>(result) = stats();
    result.stop_reason = reason;
    return result;
}

// Each worker keeps up to batch_size simulations in flight: it descends that
// many paths (each parked on its virtual loss), then submits all their leaves
// to the evaluation queue in one round-trip. This is what makes evaluation
// batches form — and it amortizes the queue handshake, which would otherwise
// dominate with a cheap evaluator.
void Search::worker_loop(const SearchLimits& limits) {
    const int max_in_flight = std::max(config_.batch_size, 1);
    std::vector<std::vector<uint32_t>> paths;  // paths[i] belongs to boards[i]
    std::vector<core::Board> boards;
    std::vector<float> values;
    bool tickets_exhausted = false;

    while (!stop_workers_.load(std::memory_order_relaxed) && !tickets_exhausted) {
        paths.clear();
        boards.clear();
        for (int k = 0; k < max_in_flight; ++k) {
            // A ticket is one simulation start; with a simulation cap,
            // exactly max_simulations tickets are handed out in total.
            if (limits.max_simulations >= 0 &&
                tickets_.fetch_add(1, std::memory_order_relaxed) >= limits.max_simulations) {
                tickets_exhausted = true;
                break;
            }
            descend(paths, boards);
        }

        if (!boards.empty()) {
            values.assign(boards.size(), 0.0f);
            queue_.evaluate(boards, values);
            for (size_t i = 0; i < boards.size(); ++i) {
                // The evaluator scores for the side to move = the opponent of
                // the player who moved into the leaf.
                backprop(paths[i], 1.0f - values[i]);
                simulations_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

SearchStats Search::stats() const {
    if (!tree_) return {};
    const Tree& tree = *tree_;
    const Node& root = tree[tree.root()];

    SearchStats s;
    s.simulations = simulations_.load(std::memory_order_relaxed);
    s.nodes = tree.size();
    s.elapsed_ms = elapsed_ms();

    // Root statistics are stored from the perspective of the player who moved
    // into the root; flip to the side to move.
    const uint32_t visits = root.visits.load(std::memory_order_relaxed);
    const double q =
        visits ? root.value_sum.load(std::memory_order_relaxed) / visits : 0.5;
    s.root_value = float(1.0 - q);
    s.root_cp = int(std::lround(eval::win_prob_to_centipawns(s.root_value)));

    // Principal variation: follow the most-visited child from the root.
    uint32_t index = tree.root();
    for (int depth = 0; depth < 20; ++depth) {
        const Node& node = tree[index];
        if (node.expand_state.load(std::memory_order_acquire) != ExpandState::EXPANDED ||
            node.num_children == 0)
            break;
        uint32_t best_index = 0, best_visits = 0;
        for (uint32_t i = 0; i < node.num_children; ++i) {
            const uint32_t child_visits =
                tree[node.first_child + i].visits.load(std::memory_order_relaxed);
            if (child_visits > best_visits) {
                best_visits = child_visits;
                best_index = node.first_child + i;
            }
        }
        if (best_visits == 0) break;
        s.pv.push_back(tree[best_index].move.uci());
        index = best_index;
    }
    if (!s.pv.empty()) s.best_move = s.pv.front();
    return s;
}

int64_t Search::elapsed_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - started_at_)
        .count();
}

}  // namespace mcts

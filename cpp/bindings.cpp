#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/board.h"
#include "core/movegen.h"
#include "eval/material.h"
#include "mcts/search.h"

namespace py = pybind11;
using core::Board;
using core::Move;

namespace {

// Composition root for the pybind boundary: one Engine = one evaluator + one
// Search. Python talks FENs and stats structs; nothing per-node crosses.
class Engine {
public:
    explicit Engine(const mcts::SearchConfig& config) : search_(config, evaluator_) {}

    void set_position(const std::string& fen) { search_.set_position(Board(fen)); }
    void advance(const std::string& uci) { search_.advance(Move::from_uci(uci)); }
    mcts::TreeSnapshot snapshot(uint32_t min_visits, int max_depth) const {
        return search_.snapshot(min_visits, max_depth);
    }
    mcts::TreeView tree_view(uint32_t max_nodes, uint32_t min_visits,
                             const std::vector<std::string>& root_path) const {
        std::vector<Move> path;
        for (const std::string& uci : root_path) path.push_back(Move::from_uci(uci));
        return search_.tree_view(max_nodes, min_visits, path);
    }
    mcts::SearchResult search(const mcts::SearchLimits& limits) { return search_.run(limits); }
    void start(const mcts::SearchLimits& limits) { search_.start(limits); }
    mcts::SearchResult stop() { return search_.stop(); }
    bool running() const { return search_.running(); }
    mcts::SearchStats stats() const { return search_.stats(); }
    void request_stop() { search_.request_stop(); }

private:
    eval::MaterialEvaluator evaluator_;
    mcts::Search search_;
};

}  // namespace

PYBIND11_MODULE(_mcts, m) {
    m.doc() = "MCTS search core (C++). See DESIGN.md section 5 for the boundary rules.";

    m.def("version", [] { return "0.1.0"; });

    // FEN-based helpers for cross-validating the C++ rules against
    // python-chess (tests/python/test_movegen.py, test_perft.py). Not used
    // by the search itself.
    m.def("legal_moves", [](const std::string& fen) {
        std::vector<std::string> out;
        for (Move mv : core::generate_legal(Board(fen))) out.push_back(mv.uci());
        return out;
    });

    m.def("apply_move", [](const std::string& fen, const std::string& uci) {
        Board board(fen);
        Move mv = Move::from_uci(uci);
        for (Move legal : core::generate_legal(board)) {
            if (legal == mv) {
                board.apply(mv);
                return board.fen();
            }
        }
        throw std::invalid_argument("illegal move: " + uci);
    });

    m.def(
        "perft",
        [](const std::string& fen, int depth) { return core::perft(Board(fen), depth); },
        py::call_guard<py::gil_scoped_release>());

    py::class_<mcts::SearchConfig>(m, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("workers", &mcts::SearchConfig::workers)
        .def_readwrite("batch_size", &mcts::SearchConfig::batch_size)
        .def_readwrite("max_nodes", &mcts::SearchConfig::max_nodes)
        .def_readwrite("seed", &mcts::SearchConfig::seed);

    py::class_<mcts::SearchLimits>(m, "SearchLimits")
        .def(py::init<>())
        .def_readwrite("max_time_ms", &mcts::SearchLimits::max_time_ms)
        .def_readwrite("max_simulations", &mcts::SearchLimits::max_simulations)
        .def_readwrite("convergence_window", &mcts::SearchLimits::convergence_window)
        .def_readwrite("convergence_cp_threshold",
                       &mcts::SearchLimits::convergence_cp_threshold)
        .def_readwrite("c_puct", &mcts::SearchLimits::c_puct)
        .def_readwrite("virtual_loss", &mcts::SearchLimits::virtual_loss);

    py::class_<mcts::SearchStats>(m, "SearchStats")
        .def_readonly("simulations", &mcts::SearchStats::simulations)
        .def_readonly("nodes", &mcts::SearchStats::nodes)
        .def_readonly("root_value", &mcts::SearchStats::root_value)
        .def_readonly("root_cp", &mcts::SearchStats::root_cp)
        .def_readonly("best_move", &mcts::SearchStats::best_move)
        .def_readonly("pv", &mcts::SearchStats::pv)
        .def_readonly("elapsed_ms", &mcts::SearchStats::elapsed_ms);

    py::class_<mcts::SearchResult, mcts::SearchStats>(m, "SearchResult")
        .def_readonly("stop_reason", &mcts::SearchResult::stop_reason);

    py::class_<mcts::TreeView>(m, "TreeView")
        .def_readonly("parent", &mcts::TreeView::parent)
        .def_readonly("move", &mcts::TreeView::move)
        .def_readonly("visits", &mcts::TreeView::visits)
        .def_readonly("q", &mcts::TreeView::q)
        .def_readonly("prior", &mcts::TreeView::prior)
        .def_readonly("children_total", &mcts::TreeView::children_total);

    py::class_<mcts::TreeSnapshot>(m, "TreeSnapshot")
        .def_readonly("fens", &mcts::TreeSnapshot::fens)
        .def_readonly("visit_counts", &mcts::TreeSnapshot::visit_counts)
        .def_readonly("values", &mcts::TreeSnapshot::values)
        .def_readonly("moves", &mcts::TreeSnapshot::moves)
        .def_readonly("child_visits", &mcts::TreeSnapshot::child_visits);

    py::class_<Engine>(m, "Engine")
        .def(py::init<const mcts::SearchConfig&>())
        .def("set_position", &Engine::set_position)
        .def("advance", &Engine::advance)
        .def("snapshot", &Engine::snapshot, py::arg("min_visits"), py::arg("max_depth"))
        .def("tree_view", &Engine::tree_view, py::arg("max_nodes"), py::arg("min_visits"),
             py::arg("root_path"), py::call_guard<py::gil_scoped_release>())
        .def("search", &Engine::search, py::call_guard<py::gil_scoped_release>())
        .def("start", &Engine::start)
        .def("stop", &Engine::stop, py::call_guard<py::gil_scoped_release>())
        .def("running", &Engine::running)
        .def("stats", &Engine::stats)
        .def("request_stop", &Engine::request_stop);
}

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "core/board.h"
#include "core/movegen.h"
#include "eval/encode.h"
#include "eval/material.h"
#include "mcts/search.h"

namespace py = pybind11;
using core::Board;
using core::Move;

namespace {

// Evaluator bridge to a Python callback:
//   callback(planes: float32 [N, 19, 8, 8]) -> (values: float32 [N], logits: float32 [N, 4672])
// values are win probabilities in [0, 1] for the side to move of each
// (canonically encoded) position. This is the one deliberate GIL-touching
// point in the whole search (DESIGN.md section 5): workers never see
// Python, only this class does, once per batch, on the single evaluator
// thread inside EvalQueue.
class PyEvaluator : public eval::Evaluator {
public:
    explicit PyEvaluator(py::object callback) : callback_(std::move(callback)) {}

    void evaluate(std::span<const eval::EvalRequest> batch) override {
        const size_t n = batch.size();

        // 1. Encode outside the GIL.
        input_.resize(n * eval::PLANES * 64);
        for (size_t i = 0; i < n; ++i)
            eval::encode_planes(
                *batch[i].board,
                std::span<float>(&input_[i * eval::PLANES * 64], eval::PLANES * 64ul));

        // 2. One GIL scope per batch: call Python, copy plain floats out.
        //    Every py:: object must be constructed AND destroyed inside this
        //    block — destructors need the GIL too.
        values_.assign(n, 0.5f);
        gathered_.assign(n, {});
        {
            py::gil_scoped_acquire gil;
            try {
                py::array_t<float> planes({py::ssize_t(n), py::ssize_t(eval::PLANES),
                                           py::ssize_t(8), py::ssize_t(8)},
                                          input_.data());  // copies
                py::tuple out = callback_(planes);
                auto values = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
                    out[0]);
                auto logits = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
                    out[1]);
                auto v = values.unchecked<1>();
                auto l = logits.unchecked<2>();
                for (size_t i = 0; i < n; ++i) {
                    values_[i] = v(py::ssize_t(i));
                    gathered_[i].reserve(batch[i].moves.size());
                    for (core::Move mv : batch[i].moves)
                        gathered_[i].push_back(
                            l(py::ssize_t(i), eval::move_index(*batch[i].board, mv)));
                }
            } catch (const std::exception& e) {
                // A broken callback must not take down the search thread:
                // report once, fall back to neutral values + uniform priors.
                // Deliberately catching std::exception, not just
                // py::error_already_set: a callback that raises in Python
                // throws error_already_set (itself std::exception-derived),
                // but one that returns the wrong shape/dtype throws a plain
                // C++ cast/domain error from py::cast/unchecked<N>() below —
                // both must be survivable.
                py::print("PyEvaluator callback failed:", e.what());
                for (auto& g : gathered_) g.clear();
            }
        }

        // 3. Outside the GIL: softmax per position over the legal subset.
        for (size_t i = 0; i < n; ++i) {
            *batch[i].value_out = values_[i];
            write_priors(batch[i], gathered_[i]);
        }
    }

private:
    // Numerically stable softmax of `gathered` (the net's logits at the
    // legal moves) into req.priors_out; uniform if the callback failed
    // (empty gathered) or returned nothing to gather (empty moves is a
    // no-op — a value-only request has no priors_out to fill).
    static void write_priors(const eval::EvalRequest& req, const std::vector<float>& gathered) {
        if (req.moves.empty()) return;
        if (gathered.empty()) {
            std::fill(req.priors_out.begin(), req.priors_out.end(),
                      1.0f / float(req.moves.size()));
            return;
        }
        const float max_logit = *std::max_element(gathered.begin(), gathered.end());
        float sum = 0.0f;
        for (size_t i = 0; i < gathered.size(); ++i) {
            const float e = std::exp(gathered[i] - max_logit);
            req.priors_out[i] = e;
            sum += e;
        }
        for (float& p : req.priors_out) p /= sum;
    }

    py::object callback_;
    // Scratch buffers: this runs on the single evaluator thread, so reusing
    // them across batches avoids reallocating every call.
    std::vector<float> input_;
    std::vector<float> values_;
    std::vector<std::vector<float>> gathered_;
};

// Composition root for the pybind boundary: one Engine = one evaluator + one
// Search. Python talks FENs and stats structs; nothing per-node crosses.
class Engine {
public:
    // evaluator = None -> the built-in material heuristic; otherwise a
    // Python callback wrapped in PyEvaluator (see PyEvaluator above).
    Engine(const mcts::SearchConfig& config, py::object evaluator)
        : evaluator_(evaluator.is_none()
                         ? std::unique_ptr<eval::Evaluator>(std::make_unique<eval::MaterialEvaluator>())
                         : std::make_unique<PyEvaluator>(std::move(evaluator))),
          search_(config, *evaluator_) {}

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
    // GIL shutdown rule: destroying Engine joins the evaluator thread (via
    // ~Search -> ~EvalQueue), which may need the GIL to finish its current
    // batch if evaluator_ is a PyEvaluator. If a search is still running and
    // the caller holds the GIL, that deadlocks — so a search must be
    // stopped before the Engine is dropped (enforced Python-side in
    // engine.py's Engine.close()). Declared before search_ so it is
    // destroyed after: the queue thread must find it alive until ~Search
    // completes.
    std::unique_ptr<eval::Evaluator> evaluator_;
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

    // Position/move encoding (DESIGN-M6.md section 3), exposed for the
    // training pipeline (planes/policy targets recomputed at training time
    // through these — never duplicated in Python) and for tests, which
    // cross-check against an independent python-chess reference.
    m.attr("PLANES") = eval::PLANES;
    m.attr("POLICY_SIZE") = eval::POLICY_SIZE;

    m.def("encode_planes", [](const std::string& fen) {
        py::array_t<float> out({eval::PLANES, 8, 8});
        eval::encode_planes(Board(fen),
                            std::span<float>(out.mutable_data(), size_t(eval::PLANES) * 64));
        return out;
    });

    m.def("move_indices", [](const std::string& fen, const std::vector<std::string>& ucis) {
        Board board(fen);
        std::vector<int> out;
        for (const std::string& u : ucis) out.push_back(eval::move_index(board, Move::from_uci(u)));
        return out;
    });

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
        .def_readwrite("virtual_loss", &mcts::SearchLimits::virtual_loss)
        .def_readwrite("root_noise_eps", &mcts::SearchLimits::root_noise_eps)
        .def_readwrite("root_dirichlet_alpha", &mcts::SearchLimits::root_dirichlet_alpha);

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
        .def(py::init<const mcts::SearchConfig&, py::object>(), py::arg("config"),
             py::arg("evaluator") = py::none())
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

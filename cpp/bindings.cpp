#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/board.h"
#include "core/movegen.h"

namespace py = pybind11;
using core::Board;
using core::Move;

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
}

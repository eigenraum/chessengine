#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_mcts, m) {
    m.doc() = "MCTS search core (C++). See DESIGN.md section 5 for the boundary rules.";

    // Build-plumbing check for M1; the Engine class arrives with M3.
    m.def("version", [] { return "0.1.0"; });
}

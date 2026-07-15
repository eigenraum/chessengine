cmake_minimum_required(VERSION 3.26)
project(chessengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_mcts
    cpp/bindings.cpp
    cpp/core/board.cpp
    cpp/core/movegen.cpp
    cpp/eval/encode.cpp
    cpp/eval/material.cpp
    cpp/mcts/search.cpp
    cpp/mcts/tree.cpp
)
target_include_directories(_mcts PRIVATE cpp)

install(TARGETS _mcts LIBRARY DESTINATION chessengine)

# Standalone stress driver for validating the parallel search under
# ThreadSanitizer (not part of the Python wheel):
#   cmake -S . -B build/tsan -DCHESSENGINE_TSAN=ON \
#         -Dpybind11_DIR=$(uv run python -m pybind11 --cmakedir)
#   cmake --build build/tsan --target search_stress
option(CHESSENGINE_TSAN "Build the ThreadSanitizer search stress driver" OFF)
if(CHESSENGINE_TSAN)
    add_executable(search_stress
        cpp/tools/search_stress.cpp
        cpp/core/board.cpp
        cpp/core/movegen.cpp
        cpp/eval/encode.cpp
        cpp/eval/material.cpp
        cpp/mcts/search.cpp
        cpp/mcts/tree.cpp
    )
    target_include_directories(search_stress PRIVATE cpp)
    target_compile_options(search_stress PRIVATE -fsanitize=thread -g -O1)
    target_link_options(search_stress PRIVATE -fsanitize=thread)
endif()

cmake_minimum_required(VERSION 3.26)
project(chessengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_mcts
    cpp/bindings.cpp
)

install(TARGETS _mcts LIBRARY DESTINATION chessengine)

// Parallel-search stress driver: runs multi-worker searches on a few
// positions. Build with -DCHESSENGINE_TSAN=ON and run under ThreadSanitizer
// to validate the lock-free tree updates; run a plain build for a quick
// throughput check.
//
//   ./search_stress [workers] [simulations]

#include <cstdio>
#include <cstdlib>

#include "core/board.h"
#include "eval/material.h"
#include "mcts/search.h"

int main(int argc, char** argv) {
    const int workers = argc > 1 ? std::atoi(argv[1]) : 8;
    const int simulations = argc > 2 ? std::atoi(argv[2]) : 50000;

    const char* fens[] = {
        core::Board::START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // kiwipete
        "k7/8/K7/8/8/8/8/7R w - - 0 1",  // mate in 1: heavy contention on one child
    };

    eval::MaterialEvaluator evaluator;
    mcts::SearchConfig config;
    config.workers = workers;
    mcts::Search search(config, evaluator);

    mcts::SearchLimits limits;
    limits.max_time_ms = 60'000;
    limits.max_simulations = simulations;
    limits.convergence_window = 0;

    std::printf("%d workers, %d simulations per position\n", workers, simulations);
    for (const char* fen : fens) {
        search.set_position(core::Board(fen));
        mcts::SearchResult result = search.run(limits);
        std::printf("%-70s best %-6s value %.3f  %8llu sims  %9llu nodes  %5lld ms\n",
                    fen, result.best_move.c_str(), double(result.root_value),
                    (unsigned long long)result.simulations,
                    (unsigned long long)result.nodes, (long long)result.elapsed_ms);
    }

    // Exercise the async interrupt path too.
    search.set_position(core::Board(fens[0]));
    limits.max_simulations = -1;
    search.start(limits);
    mcts::SearchResult result = search.stop();
    std::printf("interrupt: %s after %llu sims\n", result.stop_reason.c_str(),
                (unsigned long long)result.simulations);
    return 0;
}

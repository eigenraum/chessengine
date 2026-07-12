#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "eval/evaluator.h"

namespace mcts {

// Funnels leaf evaluations from all search workers through one evaluator
// thread, in batches (DESIGN.md section 4.3). Workers block in evaluate();
// that is fine because their virtual loss is already applied, so other
// workers keep exploring elsewhere. Batches form naturally without a flush
// timeout: whatever accumulated while the evaluator was busy is drained
// together, up to batch_size.
class EvalQueue {
public:
    EvalQueue(eval::Evaluator& evaluator, size_t batch_size)
        : evaluator_(evaluator),
          batch_size_(batch_size),
          thread_([this] { run(); }) {}

    ~EvalQueue() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_one();
        thread_.join();
    }

    // Blocking: evaluates boards[i] into values[i] (win probability for the
    // side to move). One round-trip for the whole batch — a worker parks all
    // its in-flight simulations here at once, which is what amortizes the
    // handshake cost. Requests may be split across evaluator passes.
    void evaluate(std::span<const core::Board> boards, std::span<float> values) {
        int remaining = int(boards.size());
        std::unique_lock lock(mutex_);
        for (size_t i = 0; i < boards.size(); ++i)
            pending_.push_back({&boards[i], &values[i], &remaining});
        queue_cv_.notify_one();
        done_cv_.wait(lock, [&] { return remaining == 0; });
    }

private:
    struct Request {
        const core::Board* board;
        float* value_out;
        int* remaining;  // caller's outstanding count; guarded by mutex_
    };

    void run() {
        std::vector<Request> batch;
        std::vector<const core::Board*> boards;
        std::vector<float> values;
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                if (stopping_) return;
                size_t n = std::min(batch_size_, pending_.size());
                batch.assign(pending_.begin(), pending_.begin() + long(n));
                pending_.erase(pending_.begin(), pending_.begin() + long(n));
            }

            boards.clear();
            for (const Request& request : batch) boards.push_back(request.board);
            values.assign(batch.size(), 0.0f);
            evaluator_.evaluate(boards, values);

            {
                std::lock_guard lock(mutex_);
                for (size_t i = 0; i < batch.size(); ++i) {
                    *batch[i].value_out = values[i];
                    --*batch[i].remaining;
                }
            }
            done_cv_.notify_all();
        }
    }

    eval::Evaluator& evaluator_;
    size_t batch_size_;
    std::mutex mutex_;
    std::condition_variable queue_cv_;  // evaluator waits here for work
    std::condition_variable done_cv_;   // workers wait here for their result
    std::deque<Request> pending_;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace mcts

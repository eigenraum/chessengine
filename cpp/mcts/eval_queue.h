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

    // Blocking: evaluates each request in place (values through *value_out,
    // policy through priors_out where requested). One round-trip for the
    // whole batch — a worker parks all its in-flight simulations here at
    // once, which is what amortizes the handshake cost. Requests may be
    // split across evaluator passes.
    //
    // Lifetime: the spans and pointers inside each EvalRequest point into the
    // calling worker's stack buffers, and the worker is blocked inside this
    // call until remaining == 0 — so they outlive the evaluator call by
    // construction.
    void evaluate(std::span<const eval::EvalRequest> requests) {
        int remaining = int(requests.size());
        std::unique_lock lock(mutex_);
        for (const eval::EvalRequest& request : requests)
            pending_.push_back({request, &remaining});
        queue_cv_.notify_one();
        done_cv_.wait(lock, [&] { return remaining == 0; });
    }

private:
    struct Request {
        eval::EvalRequest req;
        int* remaining;  // caller's outstanding count; guarded by mutex_
    };

    void run() {
        std::vector<Request> batch;
        std::vector<eval::EvalRequest> requests;
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                if (stopping_) return;
                size_t n = std::min(batch_size_, pending_.size());
                batch.assign(pending_.begin(), pending_.begin() + long(n));
                pending_.erase(pending_.begin(), pending_.begin() + long(n));
            }

            requests.clear();
            for (const Request& request : batch) requests.push_back(request.req);
            evaluator_.evaluate(requests);

            {
                std::lock_guard lock(mutex_);
                for (const Request& request : batch) --*request.remaining;
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

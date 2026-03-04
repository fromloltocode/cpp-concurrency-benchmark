#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "concurrency/ws_deque.hpp"
#include "concurrency/mpmc_queue.hpp"
#include "concurrency/utils.hpp"

class WorkStealingPool {
public:
  explicit WorkStealingPool(std::size_t n_threads,
                            std::size_t local_deque_cap = 1 << 16,
                            std::size_t global_cap = 1 << 16)
      : stop_(false),
        pending_(0),
        submit_rr_(0),
        global_(global_cap) {
    if (n_threads == 0) n_threads = 1;
    if (!is_pow2(local_deque_cap) || !is_pow2(global_cap)) {
      throw std::runtime_error("capacities must be powers of two");
    }

    deques_.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
      deques_.emplace_back(std::make_unique<WSDeque<Task*>>(local_deque_cap));
    }

    workers_.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
      workers_.emplace_back([this, i]() { worker_loop(i); });
    }
  }

  ~WorkStealingPool() { shutdown(); }

  WorkStealingPool(const WorkStealingPool&) = delete;
  WorkStealingPool& operator=(const WorkStealingPool&) = delete;

  void push(std::function<void()> job) {
    if (stop_.load(std::memory_order_acquire)) return;

    Task* t = new Task{std::move(job)};

    const std::size_t n = deques_.size();
    std::size_t idx = submit_rr_.fetch_add(1, std::memory_order_relaxed) % n;

    if (deques_[idx]->push_bottom(t)) {
      pending_.fetch_add(1, std::memory_order_release);
      cv_.notify_one();
      return;
    }

    // Local deque full -> global fallback (bounded)
    while (!global_.enqueue(t)) {
      if (stop_.load(std::memory_order_acquire)) { delete t; return; }
      std::this_thread::yield();
    }
    pending_.fetch_add(1, std::memory_order_release);
    cv_.notify_one();
  }

  void shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
    cv_.notify_all();

    for (auto& th : workers_) {
      if (th.joinable()) th.join();
    }
    workers_.clear();

    // Best-effort cleanup of any remaining tasks (should be none if caller waited).
    Task* t = nullptr;
    for (auto& dq : deques_) {
      while (dq->pop_bottom(t)) { delete t; t = nullptr; }
    }
    while (global_.dequeue(t)) { delete t; t = nullptr; }
  }

private:
  struct Task {
    std::function<void()> fn;
  };

  void worker_loop(std::size_t self) {
    Task* task = nullptr;

    uint64_t rng = 0x9e3779b97f4a7c15ULL ^ (self + 1);
    auto xorshift = [&]() -> uint64_t {
      rng ^= rng >> 12;
      rng ^= rng << 25;
      rng ^= rng >> 27;
      return rng * 2685821657736338717ULL;
    };

    const std::size_t n = deques_.size();

    for (;;) {
      if (pending_.load(std::memory_order_acquire) > 0) {
        // 1) local pop
        if (deques_[self]->pop_bottom(task)) {
          pending_.fetch_sub(1, std::memory_order_acq_rel);
          if (task && task->fn) task->fn();
          delete task;
          task = nullptr;
          continue;
        }

        // 2) steal
        bool got = false;
        for (int k = 0; k < 16; ++k) { // more attempts helps on small tasks
          std::size_t victim = (std::size_t)(xorshift() % n);
          if (victim == self) continue;
          if (deques_[victim]->steal_top(task)) {
            pending_.fetch_sub(1, std::memory_order_acq_rel);
            if (task && task->fn) task->fn();
            delete task;
            task = nullptr;
            got = true;
            break;
          }
        }
        if (got) continue;

        // 3) global injector
        if (global_.dequeue(task)) {
          pending_.fetch_sub(1, std::memory_order_acq_rel);
          if (task && task->fn) task->fn();
          task = nullptr;
          continue;
        }

        std::this_thread::yield();
        continue;
      }

      if (stop_.load(std::memory_order_acquire)) return;

      std::unique_lock<std::mutex> lk(cv_m_);
      cv_.wait(lk, [&]() {
        return stop_.load(std::memory_order_acquire) ||
               pending_.load(std::memory_order_acquire) > 0;
      });

      if (stop_.load(std::memory_order_acquire) &&
          pending_.load(std::memory_order_acquire) == 0) {
        return;
      }
    }
  }

  std::atomic<bool> stop_;
  std::atomic<uint64_t> pending_;
  std::atomic<std::size_t> submit_rr_;

  std::vector<std::unique_ptr<WSDeque<Task*>>> deques_;
  MPMCQueue<Task*> global_;

  std::mutex cv_m_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
};

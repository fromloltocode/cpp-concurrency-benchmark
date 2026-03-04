#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "concurrency/ws_deque.hpp"
#include "concurrency/mpmc_queue.hpp"

static inline bool is_pow2(std::size_t x) { return x && ((x & (x - 1)) == 0); }

// Work-stealing thread pool:
// - each worker has a lock-free WSDeque (Chase–Lev)
// - submissions go to a worker's local deque (round-robin)
// - idle workers steal from others
// - optional global injector (MPMC) for overflow / external pushes
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
      deques_.emplace_back(local_deque_cap);
    }

    workers_.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
      workers_.emplace_back([this, i]() { worker_loop(i); });
    }
  }

  ~WorkStealingPool() { shutdown(); }

  WorkStealingPool(const WorkStealingPool&) = delete;
  WorkStealingPool& operator=(const WorkStealingPool&) = delete;

  // Submit a job. Tries local deque first; falls back to global injector.
  void push(std::function<void()> job) {
    if (stop_.load(std::memory_order_acquire)) return;

    const std::size_t n = deques_.size();
    std::size_t idx = submit_rr_.fetch_add(1, std::memory_order_relaxed) % n;

    if (deques_[idx].push_bottom(std::move(job))) {
      pending_.fetch_add(1, std::memory_order_release);
      cv_.notify_one();
      return;
    }

    // Local deque full -> global fallback (bounded)
    while (!global_.enqueue(job)) {
      if (stop_.load(std::memory_order_acquire)) return;
      std::this_thread::yield();
    }
    pending_.fetch_add(1, std::memory_order_release);
    cv_.notify_one();
  }

  void shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
    cv_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
    workers_.clear();
  }

private:
  void worker_loop(std::size_t self) {
    std::function<void()> job;

    // Simple RNG for victim selection (fast enough)
    uint64_t rng = 0x9e3779b97f4a7c15ULL ^ (self + 1);

    auto xorshift = [&]() -> uint64_t {
      rng ^= rng >> 12;
      rng ^= rng << 25;
      rng ^= rng >> 27;
      return rng * 2685821657736338717ULL;
    };

    const std::size_t n = deques_.size();

    for (;;) {
      // Fast path: local pop
      if (pending_.load(std::memory_order_acquire) > 0) {
        if (deques_[self].pop_bottom(job)) {
          pending_.fetch_sub(1, std::memory_order_acq_rel);
          job();
          continue;
        }

        // Steal path: try a few victims
        bool got = false;
        for (int k = 0; k < 8; ++k) { // small fixed attempts
          std::size_t victim = (std::size_t)(xorshift() % n);
          if (victim == self) continue;
          if (deques_[victim].steal_top(job)) {
            pending_.fetch_sub(1, std::memory_order_acq_rel);
            job();
            got = true;
            break;
          }
        }
        if (got) continue;

        // Global injector fallback
        if (global_.dequeue(job)) {
          pending_.fetch_sub(1, std::memory_order_acq_rel);
          job();
          continue;
        }

        // pending said >0 but we didn't find work due to contention;
        // brief yield to avoid hot spinning.
        std::this_thread::yield();
        continue;
      }

      if (stop_.load(std::memory_order_acquire)) return;

      // Sleep until there is pending work or shutdown
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

  std::vector<WSDeque<std::function<void()>>> deques_;
  MPMCQueue<std::function<void()>> global_;

  std::mutex cv_m_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
};

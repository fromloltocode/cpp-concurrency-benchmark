#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include "concurrency/utils.hpp"
#include "concurrency/mpmc_queue.hpp"



// Thread pool using a lock-free bounded MPMC queue for jobs.
// - Workers block (via CV) when no work is pending.
// - Producers block (spin/yield) when queue is full (backpressure).
class ThreadPoolMPMC {
public:
  explicit ThreadPoolMPMC(std::size_t n_threads,
                          std::size_t queue_capacity = 1 << 16)
      : q_(queue_capacity),
        stop_(false),
        pending_(0) {
    if (n_threads == 0) n_threads = 1;
    if (!is_pow2(queue_capacity)) {
      throw std::runtime_error("queue_capacity must be a power of two");
    }

    workers_.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ~ThreadPoolMPMC() { shutdown(); }

  ThreadPoolMPMC(const ThreadPoolMPMC&) = delete;
  ThreadPoolMPMC& operator=(const ThreadPoolMPMC&) = delete;

  // Submit a job. Blocks (spins/yields) if queue is full.
  void push(std::function<void()> job) {
    if (stop_.load(std::memory_order_acquire)) return;

    // Backpressure: bounded queue, so keep trying.
    // (Fast in practice; if you want, we can add exponential backoff.)
    while (!q_.enqueue(job)) {
      if (stop_.load(std::memory_order_acquire)) return;
      std::this_thread::yield();
    }

    pending_.fetch_add(1, std::memory_order_release);

    // Wake one sleeping worker.
    {
      std::lock_guard<std::mutex> lk(cv_m_);
      // nothing else needed; just sync with cv
    }
    cv_.notify_one();
  }

  void shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;

    // Wake everyone so they can observe stop_.
    cv_.notify_all();

    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
    workers_.clear();
  }

private:
  void worker_loop() {
    std::function<void()> job;

    for (;;) {
      // Try to grab work if any is pending.
      if (pending_.load(std::memory_order_acquire) > 0) {
        if (q_.dequeue(job)) {
          pending_.fetch_sub(1, std::memory_order_acq_rel);
          job();
          continue;
        }
        // pending_ said >0 but dequeue failed due to contention; retry.
        std::this_thread::yield();
        continue;
      }

      // No pending work.
      if (stop_.load(std::memory_order_acquire)) return;

      // Sleep briefly until new work arrives or shutdown happens.
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

  MPMCQueue<std::function<void()>> q_;
  std::atomic<bool> stop_;
  std::atomic<uint64_t> pending_;

  std::mutex cv_m_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
};

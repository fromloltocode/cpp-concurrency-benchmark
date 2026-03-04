#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// A straightforward thread pool with a bounded queue.
// - push() blocks when the queue is full (backpressure)
// - shutdown() wakes all threads and stops cleanly
class ThreadPool {
public:
  explicit ThreadPool(std::size_t n_threads,
                      std::size_t queue_capacity = 1 << 16)
      : cap_(queue_capacity), stop_(false) {
    if (n_threads == 0) n_threads = 1;
    workers_.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
      workers_.emplace_back([this]() { this->worker_loop(); });
    }
  }

  ~ThreadPool() { shutdown(); }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Submit a job. Blocks if queue is full.
  void push(std::function<void()> job) {
    std::unique_lock<std::mutex> lk(m_);
    not_full_.wait(lk, [&]() { return stop_.load(std::memory_order_acquire) || q_.size() < cap_; });
    if (stop_.load(std::memory_order_acquire)) return;
    q_.push_back(std::move(job));
    lk.unlock();
    not_empty_.notify_one();
  }

  void shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      // already stopped
      return;
    }
    {
      std::lock_guard<std::mutex> lk(m_);
      // nothing else to do; stop_ already set
    }
    not_empty_.notify_all();
    not_full_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
    workers_.clear();
    // Optional: clear queue
    std::lock_guard<std::mutex> lk(m_);
    q_.clear();
  }

private:
  void worker_loop() {
    while (true) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&]() {
          return stop_.load(std::memory_order_acquire) || !q_.empty();
        });

        if (stop_.load(std::memory_order_acquire) && q_.empty()) return;

        job = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
      }
      job();
    }
  }

  const std::size_t cap_;
  std::atomic<bool> stop_;
  std::mutex m_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<std::function<void()>> q_;
  std::vector<std::thread> workers_;
};

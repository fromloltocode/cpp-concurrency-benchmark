#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>

#include "concurrency/ws_deque.hpp"      // WSDeque<T*> (atomic pointer slots version recommended)
#include "concurrency/mpmc_queue.hpp"    // MPMCQueue<T*>
#include "concurrency/utils.hpp"         // is_pow2

class WorkStealingPool {
public:
  explicit WorkStealingPool(std::size_t n_threads,
                            std::size_t local_deque_cap = 1 << 16,
                            std::size_t global_cap      = 1 << 16)
      : stop_(false),
        submit_rr_(0),
        inflight_(0),
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

  void push(std::function<void()> fn) {
    if (stop_.load(std::memory_order_acquire)) return;
  
    Task* t = new Task{};
    t->fn = std::move(fn);
  
    while (!global_.enqueue(t)) {
      if (stop_.load(std::memory_order_acquire)) { delete t; return; }
      std::this_thread::yield();
    }
  
    inflight_.fetch_add(1, std::memory_order_release);
    cv_.notify_one();
  }

  void wait_idle() {
    std::unique_lock<std::mutex> lk(idle_m_);
    idle_cv_.wait(lk, [&]() {
      return inflight_.load(std::memory_order_acquire) == 0;
    });
  }

  bool wait_idle_for(std::chrono::milliseconds dur) {
    std::unique_lock<std::mutex> lk(idle_m_);
    return idle_cv_.wait_for(lk, dur, [&]() {
      return inflight_.load(std::memory_order_acquire) == 0;
    });
  }

  uint64_t inflight() const {
    return inflight_.load(std::memory_order_acquire);
  }

  // Debug helper: prints inflight, and (optionally) deque approximate sizes if available.
  void debug_dump() const {
    std::cerr << "[WS] inflight=" << inflight() << "\n";

    // If your WSDeque has size_approx(), uncomment these lines:
    /*
    for (size_t i = 0; i < deques_.size(); ++i) {
      std::cerr << "  dq[" << i << "] size~=" << deques_[i]->size_approx() << "\n";
    }
    */
  }

  void shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;

    cv_.notify_all();
    {
      std::lock_guard<std::mutex> lk(idle_m_);
      idle_cv_.notify_all();
    }

    for (auto& th : workers_) {
      if (th.joinable()) th.join();
    }
    workers_.clear();

    // Best-effort cleanup of leftovers (should be none if wait_idle() was used).
    Task* t = nullptr;
    for (auto& dq : deques_) {
      while (dq->pop_bottom(t)) { delete t; t = nullptr; }
    }
    while (global_.dequeue(t)) { delete t; t = nullptr; }
  }

private:
  struct Task {
    std::atomic<uint8_t> claimed{0};
    std::function<void()> fn;
  };

  void complete_one() {
    if (inflight_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lk(idle_m_);
      idle_cv_.notify_all();
    }
  }

  void run_task(Task* t) {
    if (!t) return;
  
    uint8_t expected = 0;
    if (!t->claimed.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
      // duplicate delivery; drop it but still count it as completed
      complete_one();
      return;
    }
  
    if (t->fn) t->fn();
    delete t;
    complete_one();
  }

  void worker_loop(std::size_t self) {
    const std::size_t n = deques_.size();

    for (;;) {
      // 1) local pop
      Task* task = nullptr;
      if (deques_[self]->pop_bottom(task)) {
        run_task(task);
        continue;
      }

      // 2) deterministic steal sweep
      for (std::size_t off = 1; off < n && !task; ++off) {
        std::size_t victim = (self + off) % n;
        deques_[victim]->steal_top(task);
      }
      if (task) {
        run_task(task);
        continue;
      }

      // 3) global injector
      if (global_.dequeue(task)) {
        // prefer to execute locally via deque for LIFO locality
        if (!deques_[self]->push_bottom(task)) {
          // if local full, just run it
          run_task(task);
        }
        continue;
      }

      // 4) nothing found
      if (stop_.load(std::memory_order_acquire)) return;

      // Sleep until either stop or some work is inflight (hint).
      std::unique_lock<std::mutex> lk(cv_m_);
      cv_.wait(lk, [&]() {
        return stop_.load(std::memory_order_acquire) ||
               inflight_.load(std::memory_order_acquire) > 0;
      });
      if (stop_.load(std::memory_order_acquire)) return;
    }
  }

  std::atomic<bool> stop_;
  std::atomic<std::size_t> submit_rr_;
  std::atomic<uint64_t> inflight_;

  std::vector<std::unique_ptr<WSDeque<Task*>>> deques_;
  MPMCQueue<Task*> global_;

  std::mutex cv_m_;
  std::condition_variable cv_;

  std::mutex idle_m_;
  std::condition_variable idle_cv_;

  std::vector<std::thread> workers_;
};
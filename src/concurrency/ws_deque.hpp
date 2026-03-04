#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

// Fixed-capacity Chase–Lev work-stealing deque.
// - Single owner thread calls push_bottom/pop_bottom.
// - Any thread may call steal_top.
//
// Capacity must be a power of two.
// This version is intentionally fixed-size (no resize) to keep it tight and correct.
template <typename T>
class WSDeque {
public:
  explicit WSDeque(std::size_t capacity_pow2)
      : cap_(capacity_pow2), mask_(capacity_pow2 - 1), buf_(capacity_pow2) {
    // require power-of-two capacity
    if (cap_ == 0 || (cap_ & (cap_ - 1)) != 0) {
      throw std::runtime_error("WSDeque capacity must be power of two");
    }
    top_.store(0, std::memory_order_relaxed);
    bottom_.store(0, std::memory_order_relaxed);
  }

  WSDeque(const WSDeque&) = delete;
  WSDeque& operator=(const WSDeque&) = delete;

  // Owner-only. Returns false if full.
  bool push_bottom(T v) {
    size_t b = bottom_.load(std::memory_order_relaxed);
    size_t t = top_.load(std::memory_order_acquire);
    if (b - t >= cap_) return false; // full

    buf_[b & mask_] = std::move(v);
    // publish element before moving bottom
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
    return true;
  }

  // Owner-only. Returns false if empty.
  bool pop_bottom(T& out) {
    size_t b = bottom_.load(std::memory_order_relaxed);
    if (b == 0) return false;
    b = b - 1;
    bottom_.store(b, std::memory_order_relaxed);

    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t t = top_.load(std::memory_order_relaxed);

    if (t <= b) {
      // non-empty
      out = std::move(buf_[b & mask_]);
      if (t == b) {
        // last element: race with stealers
        if (!top_.compare_exchange_strong(t, t + 1,
                                         std::memory_order_seq_cst,
                                         std::memory_order_relaxed)) {
          // lost race: stolen
          // restore bottom to t+1 (empty)
          bottom_.store(b + 1, std::memory_order_relaxed);
          return false;
        }
        // won race: deque now empty
        bottom_.store(b + 1, std::memory_order_relaxed);
      }
      return true;
    } else {
      // empty: restore bottom
      bottom_.store(t, std::memory_order_relaxed);
      return false;
    }
  }

  // Any thread. Returns false if empty or lost race.
  bool steal_top(T& out) {
    size_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t b = bottom_.load(std::memory_order_acquire);

    if (t < b) {
      // non-empty
      out = std::move(buf_[t & mask_]);
      if (top_.compare_exchange_strong(t, t + 1,
                                      std::memory_order_seq_cst,
                                      std::memory_order_relaxed)) {
        return true;
      }
      return false;
    }
    return false;
  }

  // Approximate size (debug/metrics)
  std::size_t size_approx() const {
    size_t t = top_.load(std::memory_order_acquire);
    size_t b = bottom_.load(std::memory_order_acquire);
    return (b >= t) ? (b - t) : 0;
  }

private:
  const std::size_t cap_;
  const std::size_t mask_;
  std::vector<T> buf_;

  alignas(64) std::atomic<size_t> top_{0};
  alignas(64) std::atomic<size_t> bottom_{0};
};

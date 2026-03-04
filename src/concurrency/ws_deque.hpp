#pragma once
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Correct Chase–Lev work-stealing deque for POINTER payloads.
// - Single owner: push_bottom / pop_bottom
// - Any thread: steal_top
// Slots are atomic pointers, cleared on successful claim.
template <typename T>
class WSDeque {
  static_assert(std::is_pointer_v<T>, "WSDeque<T> expects T to be a pointer type in this project.");

public:
  explicit WSDeque(std::size_t capacity_pow2)
      : cap_(capacity_pow2),
        mask_(capacity_pow2 - 1),
        buf_(capacity_pow2) {
    if (cap_ == 0 || (cap_ & (cap_ - 1)) != 0) {
      throw std::runtime_error("WSDeque capacity must be power of two");
    }
    for (auto& a : buf_) a.store(nullptr, std::memory_order_relaxed);
    top_.store(0, std::memory_order_relaxed);
    bottom_.store(0, std::memory_order_relaxed);
  }

  WSDeque(const WSDeque&) = delete;
  WSDeque& operator=(const WSDeque&) = delete;

  bool push_bottom(T v) {
    size_t b = bottom_.load(std::memory_order_relaxed);
    size_t t = top_.load(std::memory_order_acquire);
    if (b - t >= cap_) return false; // full

    buf_[b & mask_].store(v, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
    return true;
  }

  bool pop_bottom(T& out) {
    size_t b = bottom_.load(std::memory_order_relaxed);
    if (b == 0) return false;

    b = b - 1;
    bottom_.store(b, std::memory_order_relaxed);

    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t t = top_.load(std::memory_order_relaxed);

    if (t <= b) {
      // non-empty
      T v = buf_[b & mask_].load(std::memory_order_acquire);

      if (t == b) {
        // last item: must race thieves
        size_t expected = t;
        if (!top_.compare_exchange_strong(expected, t + 1,
                                         std::memory_order_seq_cst,
                                         std::memory_order_relaxed)) {
          // stolen by someone else
          bottom_.store(t + 1, std::memory_order_relaxed);
          return false;
        }
        // owner won last-item race: restore bottom to empty state
        bottom_.store(t + 1, std::memory_order_relaxed);
      }

      // claim the slot (prevents duplicates)
      buf_[b & mask_].store(nullptr, std::memory_order_release);

      out = v;
      return v != nullptr;
    } else {
      // empty
      bottom_.store(t, std::memory_order_relaxed);
      return false;
    }
  }

  bool steal_top(T& out) {
    size_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t b = bottom_.load(std::memory_order_acquire);

    if (t < b) {
      T v = buf_[t & mask_].load(std::memory_order_acquire);
      size_t expected = t;
      if (top_.compare_exchange_strong(expected, t + 1,
                                      std::memory_order_seq_cst,
                                      std::memory_order_relaxed)) {
        // claim slot
        buf_[t & mask_].store(nullptr, std::memory_order_release);
        out = v;
        return v != nullptr;
      }
    }
    return false;
  }

private:
  const std::size_t cap_;
  const std::size_t mask_;
  std::vector<std::atomic<T>> buf_;

  alignas(64) std::atomic<size_t> top_{0};
  alignas(64) std::atomic<size_t> bottom_{0};
};

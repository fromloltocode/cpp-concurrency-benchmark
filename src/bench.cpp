#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <iomanip>
#include <condition_variable>

#include "thread_pool.hpp"
#include "thread_pool_mpmc.hpp"

using Clock = std::chrono::steady_clock;

static inline uint64_t ns_since(const Clock::time_point& start) {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

struct Result {
  const char* name;
  int threads;
  uint64_t ops;
  uint64_t ns;
};

static void print_result(const Result& r) {
  double sec = (double)r.ns / 1e9;
  double ops_sec = sec > 0 ? (double)r.ops / sec : 0.0;
  std::cout << std::left << std::setw(32) << r.name
            << " threads=" << std::setw(2) << r.threads
            << " time=" << std::setw(8) << std::fixed << std::setprecision(3) << sec << "s"
            << " ops=" << r.ops
            << " ops/s=" << (uint64_t)ops_sec
            << "\n";
}

// -----------------------------
// 1) CPU-bound scaling
// -----------------------------
static uint64_t cpu_work(uint64_t iters) {
  uint64_t x = 0x9e3779b97f4a7c15ULL;
  for (uint64_t i = 0; i < iters; i++) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x *= 2685821657736338717ULL;
  }
  return x;
}

static Result bench_cpu_scale(int threads, uint64_t total_iters) {
  uint64_t per = total_iters / (uint64_t)threads;
  std::vector<std::thread> ts;
  ts.reserve(threads);

  std::atomic<uint64_t> sink{0};

  auto start = Clock::now();
  for (int t = 0; t < threads; t++) {
    ts.emplace_back([&, per]() {
      uint64_t v = cpu_work(per);
      sink.fetch_add(v, std::memory_order_relaxed);
    });
  }
  for (auto& th : ts) th.join();
  uint64_t ns = ns_since(start);

  return {"cpu_scale (mix loop)", threads, per * (uint64_t)threads, ns};
}

// -----------------------------
// 2) Atomic increment contention
// -----------------------------
static Result bench_atomic_inc(int threads, uint64_t iters_per_thread) {
  std::atomic<uint64_t> x{0};
  std::vector<std::thread> ts;
  ts.reserve(threads);

  auto start = Clock::now();
  for (int t = 0; t < threads; t++) {
    ts.emplace_back([&, iters_per_thread]() {
      for (uint64_t i = 0; i < iters_per_thread; i++) {
        x.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& th : ts) th.join();
  uint64_t ns = ns_since(start);

  return {"atomic_fetch_add (contended)", threads, iters_per_thread * (uint64_t)threads, ns};
}

// -----------------------------
// 3) Mutex increment contention
// -----------------------------
static Result bench_mutex_inc(int threads, uint64_t iters_per_thread) {
  uint64_t x = 0;
  std::mutex m;
  std::vector<std::thread> ts;
  ts.reserve(threads);

  auto start = Clock::now();
  for (int t = 0; t < threads; t++) {
    ts.emplace_back([&, iters_per_thread]() {
      for (uint64_t i = 0; i < iters_per_thread; i++) {
        std::lock_guard<std::mutex> lk(m);
        x++;
      }
    });
  }
  for (auto& th : ts) th.join();
  uint64_t ns = ns_since(start);

  return {"mutex++ (contended)", threads, iters_per_thread * (uint64_t)threads, ns};
}

// -----------------------------
// 4) SPSC ring buffer throughput
// -----------------------------
template <typename T, size_t CAP>
struct SPSC {
  static_assert((CAP & (CAP - 1)) == 0, "CAP must be power of 2");
  alignas(64) std::atomic<size_t> head{0};
  alignas(64) std::atomic<size_t> tail{0};
  alignas(64) T buf[CAP];

  bool push(const T& v) {
    size_t h = head.load(std::memory_order_relaxed);
    size_t t = tail.load(std::memory_order_acquire);
    if (((h + 1) & (CAP - 1)) == t) return false;
    buf[h] = v;
    head.store((h + 1) & (CAP - 1), std::memory_order_release);
    return true;
  }

  bool pop(T& out) {
    size_t t = tail.load(std::memory_order_relaxed);
    size_t h = head.load(std::memory_order_acquire);
    if (t == h) return false;
    out = buf[t];
    tail.store((t + 1) & (CAP - 1), std::memory_order_release);
    return true;
  }
};

static Result bench_spsc(uint64_t messages) {
  constexpr size_t CAP = 1 << 16;
  SPSC<uint64_t, CAP> q;
  std::atomic<bool> done{false};
  std::atomic<uint64_t> recv{0};

  auto start = Clock::now();

  std::thread consumer([&]() {
    uint64_t x;
    while (!done.load(std::memory_order_acquire) || q.pop(x)) {
      if (q.pop(x)) recv.fetch_add(1, std::memory_order_relaxed);
      else std::this_thread::yield();
    }
  });

  std::thread producer([&]() {
    for (uint64_t i = 0; i < messages; i++) {
      while (!q.push(i)) std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  });

  producer.join();
  consumer.join();

  uint64_t ns = ns_since(start);
  return {"SPSC ring buffer", 2, messages, ns};
}

// -----------------------------
// 5) False sharing: padded vs unpadded
// -----------------------------
struct Counter { std::atomic<uint64_t> v{0}; };
struct alignas(64) PaddedCounter { std::atomic<uint64_t> v{0}; char pad[64 - sizeof(std::atomic<uint64_t>)]{}; };

static Result bench_false_sharing_unpadded(int threads, uint64_t iters_per_thread) {
  std::vector<Counter> cs((size_t)threads);
  std::vector<std::thread> ts;
  ts.reserve(threads);

  auto start = Clock::now();
  for (int t = 0; t < threads; t++) {
    ts.emplace_back([&, t]() {
      for (uint64_t i = 0; i < iters_per_thread; i++) {
        cs[(size_t)t].v.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& th : ts) th.join();
  uint64_t ns = ns_since(start);

  return {"false_sharing (unpadded)", threads, iters_per_thread * (uint64_t)threads, ns};
}

static Result bench_false_sharing_padded(int threads, uint64_t iters_per_thread) {
  std::vector<PaddedCounter> cs((size_t)threads);
  std::vector<std::thread> ts;
  ts.reserve(threads);

  auto start = Clock::now();
  for (int t = 0; t < threads; t++) {
    ts.emplace_back([&, t]() {
      for (uint64_t i = 0; i < iters_per_thread; i++) {
        cs[(size_t)t].v.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& th : ts) th.join();
  uint64_t ns = ns_since(start);

  return {"false_sharing (padded)", threads, iters_per_thread * (uint64_t)threads, ns};
}

// -----------------------------
// 6) Thread pool benchmarks
// -----------------------------
static Result bench_thread_pool_mutex(int pool_threads, uint64_t tasks, uint64_t work_iters_per_task) {
  ThreadPool pool((size_t)pool_threads, 1 << 16);

  std::atomic<uint64_t> remaining{tasks};
  std::mutex m;
  std::condition_variable cv;

  auto start = Clock::now();

  for (uint64_t i = 0; i < tasks; i++) {
    pool.push([&, work_iters_per_task]() {
      (void)cpu_work(work_iters_per_task);
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lk(m);
        cv.notify_one();
      }
    });
  }

  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&]() { return remaining.load(std::memory_order_acquire) == 0; });
  }

  uint64_t ns = ns_since(start);
  pool.shutdown();
  return {"thread_pool (mutex queue)", pool_threads, tasks, ns};
}

static Result bench_thread_pool_mpmc(int pool_threads, uint64_t tasks, uint64_t work_iters_per_task) {
  ThreadPoolMPMC pool((size_t)pool_threads, 1 << 16);

  std::atomic<uint64_t> remaining{tasks};
  std::mutex m;
  std::condition_variable cv;

  auto start = Clock::now();

  for (uint64_t i = 0; i < tasks; i++) {
    pool.push([&, work_iters_per_task]() {
      (void)cpu_work(work_iters_per_task);
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lk(m);
        cv.notify_one();
      }
    });
  }

  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&]() { return remaining.load(std::memory_order_acquire) == 0; });
  }

  uint64_t ns = ns_since(start);
  pool.shutdown();
  return {"thread_pool (mpmc queue)", pool_threads, tasks, ns};
}

int main() {
  const int hw = (int)std::thread::hardware_concurrency();
  const int max_threads = hw > 0 ? hw : 8;

  std::cout << "C++ Concurrency Bench\n";
  std::cout << "hardware_concurrency=" << max_threads << "\n\n";

  const uint64_t cpu_total_iters = 300000000ULL;
  const uint64_t inc_total = 40'000'000ULL;
  const uint64_t fs_iters  = 20000000ULL;
  const uint64_t spsc_msgs = 20000000ULL;

  for (int t : {1, 2, 4, 8, 16, 32}) {
    if (t > max_threads) continue;
    print_result(bench_cpu_scale(t, cpu_total_iters));
  }
  std::cout << "\n";

  for (int t : {1, 2, 4, 8, 16, 32}) {
    if (t > max_threads) continue;
    print_result(bench_atomic_inc(t, inc_total / (uint64_t)t));
  }
  std::cout << "\n";

  for (int t : {1, 2, 4, 8, 16}) {
    if (t > max_threads) continue;
    print_result(bench_mutex_inc(t, (inc_total / (uint64_t)t) / 10));
  }
  std::cout << "\n";

  print_result(bench_spsc(spsc_msgs));
  std::cout << "\n";

  for (int t : {2, 4, 8, 16, 32}) {
    if (t > max_threads) continue;
    print_result(bench_false_sharing_unpadded(t, fs_iters));
    print_result(bench_false_sharing_padded(t, fs_iters));
    std::cout << "\n";
  }

  const uint64_t tasks = 2'000'000;

  for (auto work_iters : {0ULL, 50ULL}) {
    std::cout << "\n=== thread_pool tasks=" << tasks
              << " work_iters=" << work_iters << " ===\n";
    for (int t : {1, 2, 4, 8, 16, 32}) {
      if (t > max_threads) continue;
      print_result(bench_thread_pool_mutex(t, tasks, work_iters));
      print_result(bench_thread_pool_mpmc(t, tasks, work_iters));
      std::cout << "\n";
    }
  }

  return 0;
}

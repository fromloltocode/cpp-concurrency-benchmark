# cpp-concurrency-lab

A small C++20 playground for concurrency + performance benchmarks:
- scaling (CPU-bound)
- contention (atomics vs mutex)
- queues (SPSC ring buffer)
- false sharing demos

Build:
- `make`
Run:
- `make run`


## Debug (ASAN/UBSAN)

make clean
make CXXFLAGS="-O1 -g -std=c++20 -pthread -fsanitize=address,undefined -fno-omit-frame-pointer" \
     LDFLAGS="-fsanitize=address,undefined" run

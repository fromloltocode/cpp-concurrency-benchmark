#pragma once
#include <atomic>
#include <cstddef>
#include <vector>

template<typename T>
class MPMCQueue {
public:
    explicit MPMCQueue(size_t size)
        : size_(size), mask_(size - 1), buffer_(size) {
        for (size_t i = 0; i < size_; i++) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    bool enqueue(const T& data) {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)pos;

            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }

        cell->data = data;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(T& data) {
        Cell* cell;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);

            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }

        data = cell->data;
        cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
        return true;
    }

private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data;
    };

    size_t size_;
    size_t mask_;
    std::vector<Cell> buffer_;

    std::atomic<size_t> enqueue_pos_;
    std::atomic<size_t> dequeue_pos_;
};

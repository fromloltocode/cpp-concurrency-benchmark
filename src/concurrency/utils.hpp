#pragma once
#include <cstddef>

static inline bool is_pow2(std::size_t x) {
  return x && ((x & (x - 1)) == 0);
}

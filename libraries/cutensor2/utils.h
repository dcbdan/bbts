#pragma once

#include "../cutensor/cu.h"

namespace _cutensor_utils {
  using M = std::remove_reference<decltype(cu_meta_t(0).m())>::type;

  typedef std::vector<int64_t> dims_t;
  typedef std::vector<int>     modes_t;

  int64_t product(dims_t const& dims) {
    int64_t r = 1;
    for(int64_t const& d: dims) {
      r *= d;
    }
    return r;
  }
}


#pragma once

#include <stdexcept>
#include <algorithm>

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

size_t maximum(modes_t const& ms) {
  size_t ret = 0;
  for(size_t v: ms) {
    ret = std::max(v,ret);
  }
  return ret;
}

struct castable_op_t {
  castable_op_t(int i) {
    if( i == 0) {
      cu_op = CUTENSOR_OP_ADD;
    } else if (i == 1) {
      cu_op = CUTENSOR_OP_MAX;
    } else if (i == 2) {
      cu_op = CUTENSOR_OP_MIN;
    } else if (i == 3) {
      cu_op = CUTENSOR_OP_MUL;
    } else {
      throw std::invalid_argument("is not a castable operator");
    }
  }
  castable_op_t(castable_op_t const& other): cu_op(other.cu_op) {}

  castable_op_t(): cu_op(CUTENSOR_OP_ADD) {}

  cutensorOperator_t cu_op;
};

}


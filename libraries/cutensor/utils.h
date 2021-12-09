#pragma once

#include <stdexcept>
#include <algorithm>
#include <mkl.h>

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
      mkl_op = vsAdd;
    } else if (i == 1) {
      cu_op = CUTENSOR_OP_MAX;
      mkl_op = vsFmax;
    } else if (i == 2) {
      cu_op = CUTENSOR_OP_MIN;
      mkl_op = vsFmin;
    } else if (i == 3) {
      cu_op = CUTENSOR_OP_MUL;
      mkl_op = vsMul;
    } else {
      throw std::invalid_argument("is not a castable operator");
    }
  }
  castable_op_t(castable_op_t const& other): cu_op(other.cu_op) {}

  castable_op_t(): cu_op(CUTENSOR_OP_ADD) {}

  decltype(vsAdd)* mkl_op;
  cutensorOperator_t cu_op;
};

struct unary_op_t {
  int i;
  float f;
};

void parse_uop(
  const bbts::ud_impl_t::tensor_params_t &params,
  unary_op_t& op,
  int& which) {

  op.i = params.get_raw(which++).i;
  if(op.i < 0 || op.i > 6) {
    throw std::invalid_argument("is not a valid unary op");
  }
  if(op.i == 6) {
    op.f = params.get_raw(which++).f;
  }
}

dims_t cu_shape_as_vec(cu_shape_t s) {
  dims_t ret(s.rank);
  std::copy(s.dims, s.dims + s.rank, ret.begin());
  return ret;
}


struct permute_t {
  // COLUMN MAJOR
  permute_t(dims_t dims_, modes_t ordering_, float* inn_):
    dims(dims_), ordering(ordering_), inn(inn_), ret(nullptr)
  {
    // at each iteration, put the highest dim in the last spot
    int rank = ordering.size();
    for(int r = rank-1; r > 0; --r) {
      int which = r;
      int score = ordering[r];
      for(int j = 0; j != r; ++j) {
        if(ordering[j] > score) {
          which = j;
          score = ordering[j];
        }
      }
      if(r == which) {
        continue;
      }
      // do the (batch if r < rank-1) data transpose
      swap(dims, r, which);
      // fix dims and ordering
      // example: cdabe, r = 3, which = 1, result: abcde
      int rev_r     = (rank-1) - r    ;
      int rev_which = (rank-1) - which;
      std::rotate(dims.rbegin()     + rev_r, dims.rbegin()     + rev_which, dims.rend());
      std::rotate(ordering.rbegin() + rev_r, ordering.rbegin() + rev_which, ordering.rend());
    }
  }

  void swap(
    dims_t const& dims,
    int x,
    int y)
  {
    if(x > y) {
      std::swap(x,y);
    }

    int batch_size = 1;
    int r = dims.size() - 1;
    for(; r != y; --r) {
      batch_size *= dims[r];
    }
    int ncol = 1;
    for(; r != x; --r) {
      ncol *= dims[r];
    }
    int nrow = 1;
    for(; r >= 0; --r) {
      nrow *= dims[r];
    }
    int mat_size = nrow*ncol;

    bool in_place = (ret != nullptr);
    if(!in_place) {
      ret = new float[product(dims)];
    }

    if(in_place) {
      float* ptr = ret;
      mkl_simatcopy_batch_strided(
        'c', 't',
        nrow, ncol, 1.0,
        ptr, nrow, ncol, mat_size,
        batch_size);
    } else {
      float* ptr_inn = inn;
      float* ptr_ret = ret;
      mkl_somatcopy_batch_strided(
        'c', 't',
        nrow, ncol, 1.0,
        ptr_inn, nrow, mat_size,
        ptr_ret, ncol, mat_size,
        batch_size);
    }
  }

  float* get() {
    if(ret != nullptr) {
      return ret;
    }
    return inn;
  }

  void free() {
    if(ret != nullptr) {
      delete ret;
    }
    ret = nullptr;
  }

  ~permute_t() {
    this->free();
  }

  dims_t dims;
  modes_t ordering;
  float* inn;
  float* ret;
};

}

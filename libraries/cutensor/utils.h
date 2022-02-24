#pragma once

#include <stdexcept>
#include <algorithm>
#include <mkl.h>

#include <chrono>

#include "../cutensor/cu.h"

namespace _cutensor_utils {

// These are file paths to various
// tensors that are read in by cutensor/init
std::vector<std::string> _input_files {
  "/home/daniel/data/tensors/X_1024.raw",
  "/home/daniel/data/tensors/Y_1024.raw",
  "/home/daniel/data/tensors/W1_2048.raw",
  "/home/daniel/data/tensors/W2_2048.raw"
};

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

float scalarAdd(float lhs, float rhs) { return lhs + rhs; }
float scalarMax(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }
float scalarMin(float lhs, float rhs) { return lhs < rhs ? lhs : rhs; }
float scalarMul(float lhs, float rhs) { return lhs * rhs; }

struct castable_op_t {
  castable_op_t(int i): i(i) {
    if( i == 0) {
      cu_op = CUTENSOR_OP_ADD;
      mkl_op = vsAdd;
      scalar_op = scalarAdd;
    } else if (i == 1) {
      cu_op = CUTENSOR_OP_MAX;
      mkl_op = vsFmax;
      scalar_op = scalarMax;
    } else if (i == 2) {
      cu_op = CUTENSOR_OP_MIN;
      mkl_op = vsFmin;
      scalar_op = scalarMin;
    } else if (i == 3) {
      cu_op = CUTENSOR_OP_MUL;
      mkl_op = vsMul;
      scalar_op = scalarMul;
    } else {
      throw std::invalid_argument("is not a castable operator");
    }
  }
  castable_op_t(castable_op_t const& other): cu_op(other.cu_op) {}

  castable_op_t(): cu_op(CUTENSOR_OP_ADD) {}

  cutensorOperator_t cu_op;
  decltype(vsAdd)* mkl_op;
  decltype(scalarAdd)* scalar_op;
  int i;
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

int64_t get_offset(dims_t const& dims, dims_t const& idxs) {
  int64_t s = 1;
  dims_t stride;
  for(int64_t const& d: dims) {
    stride.push_back(s);
    s *= d;
  }
  int64_t ret = 0;
  for(int r = 0; r != dims.size(); ++r){
    ret += stride[r]*idxs[r];
  }
  return ret;
}

float max_difference(int n, float* lhs, float* rhs) {
  float ret = 0.0;
  for(int i = 0; i != n; ++i) {
    float v = lhs[i] > rhs[i] ? lhs[i] - rhs[i] : rhs[i] - lhs[i];
    if(v > ret) {
      ret = v;
    }
  }
  return ret;
}

int64_t get_offset_wrt_ordering(
  dims_t const& inc_dims,
  dims_t const& inc_idxs,
  modes_t const& ordering)
{
  dims_t dims, idxs;
  for(auto m: ordering) {
    dims.push_back(inc_dims[m]);
    idxs.push_back(inc_idxs[m]);
  }
  return get_offset(dims, idxs);
}

struct for_each_index {
  for_each_index(dims_t const& dims_):
    dims(dims_)
  {}

  using op_t = std::function<void(dims_t)>;

  void operator()(op_t op) {
    if(dims.size() == 0) {
      op(dims_t());
    } else {
      dims_t index(dims.size(), 0);
      recurse(dims.size()-1, index, op);
    }
  }

  void recurse(int s, dims_t idx, op_t op) {
    if(s == 0) {
      for(int i = 0; i != dims[0]; ++i) {
        idx[0] = i;
        op(idx);
      }
    } else {
      for(int i = 0; i != dims[s]; ++i) {
        idx[s] = i;
        recurse(s-1, idx, op);
      }
    }
  }

  dims_t const& dims;
};

// TODO: permute_t has a scale method, which isn't really related to permuting..
// Rename this to something else.. Basically it is a "do these ops and only allocate
// temporary memory when required and if you do allocate, delete it on the way out"
struct permute_t {
  // COLUMN MAJOR
  permute_t(dims_t dims_, modes_t ordering_, float* inn_, bool always_in_place=false):
    dims(dims_), ordering(ordering_), inn(inn_), ret(nullptr), delete_ret(true)
  {
    if(inn == nullptr) {
      throw std::invalid_argument("input pointer cannot be null ptr");
    }
    if(always_in_place) {
      // force in place computation always and don't delete ret
      ret = inn;
      delete_ret = false;
    }
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

  bool will_delete() const {
    return delete_ret && ret != nullptr;
  }

  float* get() {
    if(ret != nullptr) {
      return ret;
    }
    return inn;
  }

  void free() {
    if(delete_ret && ret != nullptr) {
      delete[] ret;
    }
    ret = nullptr;
  }

  void scale(float v) {
    // not using an is-close comparison makes sense here, right?
    if(v != 1.0) {
      int n = product(dims);
      if(ret == nullptr) {
        ret = new float[n];
        vsMulI(n, inn, 1, &v, 0, ret, 1);
      } else {
        cblas_sscal(n, v, ret, 1);
      }
    }
  }

  ~permute_t() {
    this->free();
  }

  dims_t dims;
  modes_t ordering;
  float* inn;
  float* ret;
  bool delete_ret;
};

void inplace_permute(dims_t const& dims, modes_t const& ordering, float* ptr) {
  // call the constructor, tell it to always do in place
  permute_t p(dims, ordering, ptr, true);
}

struct cutensor_timer_t {
  cutensor_timer_t(std::string msg): msg(msg) {
    start = std::chrono::high_resolution_clock::now();
  }

  ~cutensor_timer_t() {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << std::endl << msg << ": " << duration.count() << "ms" << std::endl;
  }

  decltype(std::chrono::high_resolution_clock::now()) start;
  std::string msg;
};

}

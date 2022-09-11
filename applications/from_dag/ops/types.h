#pragma once

#include "../../../src/tensor/tensor.h"
#include "../../../src/tensor/tensor_factory.h"
#include "../../../src/ud_functions/ud_function.h"
#include "../../../src/ud_functions/udf_manager.h"

#include "print_vector.h"

#include <stdexcept>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <mutex>
#include <thread>
#include <tuple>
#include <fstream>
#include <cmath>

#include <mkl.h>

using namespace bbts;
using tensor_args_t = ud_impl_t::tensor_args_t;
using ud_impl_callable = std::function<void(const bbts::ud_impl_t::tensor_params_t &params,
                                            const tensor_args_t &_in,
                                            tensor_args_t &_out)>;

using std::vector;
using std::tuple;

// For debugging, you can turn off kernels entirely...
// Useful if trying to figure out where something breaks.

// #define CU_INIT_OFF
// #define CU_PERMUTE_OFF              // 7
// #define CU_UNARY_EW_OFF             // 4
// #define CU_BINARY_EW_OFF            // 3
// #define CU_CASTABLE_EW_OFF          // 5
// #define CU_BATCH_MATMUL_OFF         // 1
// #define CU_REDUCTION_OFF            // 2
// #define CU_EXPAND_OFF               // 6

//#define CU_BARB_REFERENCE

#define DCB01(x)
#define DCB_BEW(x) std::cout << "bew " << __LINE__ << " | " << x << std::endl

//std::mutex _dcb01_m;
//#define DCB01(x) { \
//  std::lock_guard<std::mutex> lock(_dcb01_m); \
//  std::string s(__FILE__); \
//  std::string ss(s.end() - 15, s.end()); \
//  std::cout << ss << " " << x << std::endl; \
//} char asdadasdasdasdasdasd

#define PRINTLINE //std::cout << __FILE__ << ": " << __LINE__ << std::endl

#define MAXRANK 8

//#define CU_DEBUG

#ifdef CU_DEBUG
using time_measurement_t = decltype(std::chrono::high_resolution_clock::now());
void cu_debug_write(
  time_measurement_t start_,
  time_measurement_t end_,
  std::string name)
{
  static std::mutex m;
  static std::ofstream file("cu_debug.out");

  // one line is (start, end, thread, name)
  std::thread::id id = std::this_thread::get_id();
  auto start =
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      start_.time_since_epoch()).count();
  auto end =
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_.time_since_epoch()).count();

  std::lock_guard<std::mutex> lock(m);
  file << start << "," << end << "," << id << "," << name << std::endl;
  file.flush();
}

struct cu_debug_write_t_ {
  cu_debug_write_t_(std::string name):
    name(name), start(std::chrono::high_resolution_clock::now())
  {}

  ~cu_debug_write_t_() {
    auto end = std::chrono::high_resolution_clock::now();
    cu_debug_write(start, end, name);
  }

  std::string name;
  time_measurement_t start;
};
#endif

#ifdef CU_DEBUG
#define cu_debug_write_t(name) \
  cu_debug_write_t_ Sweet_llamas_of_the_Bahamas(name)
#else
#define cu_debug_write_t(name)
#endif

struct cu_shape_t {
  int64_t rank;
  int64_t dims[MAXRANK];
};

struct cu_meta_t : public tensor_meta_t {

  int64_t& size() const {
    // which is placed at the start of the blob
    return *((int64_t*) _blob);
  }

  cu_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta
  cu_meta_t(tfid_t _id, int64_t sz) : tensor_meta_t{.fmt_id = _id} {
    this->size() = sz;
  }

  uint64_t get_data_size() const {
    uint64_t num = this->num_elem();
    return sizeof(float) * num;
  }

  uint64_t num_elem() const {
    return static_cast<uint64_t>(this->size());
  }
};

struct cu_t : public tensor_t {

  // return the meta data of the dense tensor
  cu_meta_t &meta() const {
    return *((cu_meta_t*) &_meta);
  }

  // returns the payload of the tensor
  void *data() {
    return (void*) _blob;
  }

  // return creation functions
  static tensor_creation_fs_t get_creation_fs() {
    // return the init function
    auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
      auto &t = *(cu_t *) here;
      auto &m = *(cu_meta_t * ) &_meta;
      t.meta() = m;
      return t;
    };

    // return the size
    auto size = [](const tensor_meta_t &_meta) {
      auto &m = *(cu_meta_t *) &_meta;
      return sizeof(tensor_meta_t) + m.get_data_size();
    };

    auto pnt = [](const void* here, std::stringstream &ss) {
      auto &t = *(cu_t *) here;
      ss << "Tensor[" << t.meta().num_elem() << "]" << std::endl;
    };

    // return the tensor creation functions
    return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt};
  }
};

inline float scalarAdd(float lhs, float rhs) { return lhs + rhs; }
inline float scalarMax(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }
inline float scalarMin(float lhs, float rhs) { return lhs < rhs ? lhs : rhs; }
inline float scalarMul(float lhs, float rhs) { return lhs * rhs; }

inline float aggAdd(float* data, int64_t n) {
  float ret;
  cblas_saxpy(n, 1.0, data, 1, &ret, 0);
  return ret;
};

inline float aggMax(float* data, int64_t n) { return data[cblas_isamax(n, data, 1)]; }

inline float aggMin(float* data, int64_t n) { return data[cblas_isamin(n, data, 1)]; }

inline float aggMul(float* data, int64_t n) {
  float ret = data[0];
  for(int64_t i = 1; i < n; ++i) {
    ret *= data[i];
  }
  return ret;
}

struct castable_op_t {
  castable_op_t(int i): i(i) {
    if( i == 0) {
      mkl_op = vsAdd;
      scalar_op = scalarAdd;
      agg_op = aggAdd;
    } else if (i == 1) {
      mkl_op = vsFmax;
      scalar_op = scalarMax;
      agg_op = aggMax;
    } else if (i == 2) {
      mkl_op = vsFmin;
      scalar_op = scalarMin;
      agg_op = aggMin;
    } else if (i == 3) {
      mkl_op = vsMul;
      scalar_op = scalarMul;
      agg_op = aggMul;
    } else {
      throw std::invalid_argument("is not a castable operator");
    }
  }
  castable_op_t(castable_op_t const& other): castable_op_t(other.i) {}

  castable_op_t(): castable_op_t(0) {}

  decltype(vsAdd)* mkl_op;
  decltype(scalarAdd)* scalar_op;
  decltype(aggAdd)* agg_op;
  int i;
};

struct unary_op_t {
  int i;
  float f; // this scales the output
};

int parse_uop(
  const bbts::ud_impl_t::tensor_params_t &params,
  unary_op_t& op,
  int& which) {

  op.i = params.get_raw(which++).i;
  if(op.i < 0 || op.i > 7) {
    throw std::invalid_argument("is not a valid unary op");
  }
  if(op.i == 6 || op.i == 7) {
    op.f = params.get_raw(which++).f;
  }

  return which;
}

vector<int64_t> cu_shape_as_vec(cu_shape_t s) {
  vector<int64_t> ret(s.rank);
  std::copy(s.dims, s.dims + s.rank, ret.begin());
  return ret;
}

std::ostream& operator<<(std::ostream& os, cu_shape_t const& meta) {
  os << cu_shape_as_vec(meta);
  return os;
}

int64_t product_dims(vector<int64_t> const& ds) {
  int64_t ret = 1;
  for(auto const& d: ds) {
    ret *= d;
  }
  return ret;
}

int64_t product_ints(vector<int> const& ds) {
  int64_t ret = 1;
  for(auto const& d: ds) {
    ret *= d;
  }
  return ret;
}

int parse_vector(
  bbts::ud_impl_t::tensor_params_t const& params,
  int& i,
  vector<int64_t>& ret)
{
  ret.resize(params.get_raw(i++).i);
  for(int64_t& val: ret) {
    val = params.get_raw(i++).i;
  }
  return i;
}


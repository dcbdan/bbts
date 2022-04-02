#pragma once

#include "../../../src/tensor/tensor.h"
#include "../../../src/ud_functions/ud_function.h"

#include <stdexcept>
#include <sstream>

using namespace bbts;
using tensor_args_t = ud_impl_t::tensor_args_t;
using ud_impl_callable = std::function<void(const bbts::ud_impl_t::tensor_params_t &params,
                                            const tensor_args_t &_in,
                                            tensor_args_t &_out)>;

#define PRINTLINE std::cout << __FILE__ << ": " << __LINE__ << std::endl

#define MAXRANK   4

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

//#define CU_BARB_REFERENCE

// just assume we're using CUDA_R_32F everywhere
#define SIZEOFFLOAT 4

struct cu_shape_t {
  uint32_t rank;
  int64_t dims[MAXRANK];
};

struct cu_meta_t : public tensor_meta_t {

  // returns the shape
  cu_shape_t& m() const {
    // which is placed at the start of the blob
    return *((cu_shape_t*) _blob);
  }

  cu_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta
  cu_meta_t(tfid_t _id, std::vector<int32_t> dims) : tensor_meta_t{.fmt_id = _id} {
    auto& meta = this->m();
    meta.rank = dims.size();
    for(int r = 0; r != meta.rank; ++r) {
      meta.dims[r] = dims[r];
    }
  }

  size_t get_data_size() const {
    size_t num = this->num_elem();
    return SIZEOFFLOAT * num;
  }

  size_t num_elem() const {
    auto const& meta = this->m();

    size_t num = 1;
    for(int r = 0; r != meta.rank; ++r) {
      num *= meta.dims[r];
    }

    return num;
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
      auto rank = t.meta().m().rank;
      auto dims = t.meta().m().dims;
      float* data = (float*)t.data();

      if(rank == 0) {
        ss << "scalar[" << data[0] << "]" << std::endl;
        return;
      }

      size_t n = 1;
      ss << "dims[";
      for(int r = 0; r != rank-1; ++r) {
        ss << dims[r] << ",";
        n *= dims[r];
      }
      ss << dims[rank-1] << "]" << std::endl;
      n *= dims[rank-1];

      if(rank != 2) {
        for(int i = 0; i != n; ++i) {
          ss << data[i] << " ";
        }
        ss << std::endl;
      } else {
        // cutensor is column major!
        for(int row = 0; row != dims[0]; ++row) {
          for(int col = 0; col != dims[1]; ++col) {
            int idx = row + col*dims[0];
            std::cout << data[idx] << " ";
          }
          std::cout << std::endl;
        }
      }
    };

    // return the tensor creation functions
    return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt};
  }
};

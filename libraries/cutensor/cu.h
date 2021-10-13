#pragma once

#include "../../src/tensor/tensor.h"
#include "../../src/ud_functions/ud_function.h"

using namespace bbts;
using tensor_args_t = ud_impl_t::tensor_args_t;
using ud_impl_callable = std::function<void(const bbts::ud_impl_t::tensor_params_t &params,
                                            const tensor_args_t &_in,
                                            tensor_args_t &_out)>;

#define MAXRANK   4

// just assume we're using CUDA_R_32F everywhere
#define SIZEOFFLOAT 4   
cudaDataType_t cutensor_scalar_type = CUDA_R_32F;

struct cu_meta_t : public tensor_meta_t {

  // returns the meta data struct
  auto &m() const {

    struct m {
      uint32_t rank;
      int64_t dims[MAXRANK];
    };

    // we use it as the blob
    return *((m*) _blob);
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

  size_t get_data_size() {
    auto &meta = this->m();
    
    int num = 1;
    for(int r = 0; r != meta.rank; ++r) {
      num *= meta.dims[r];
    }

    return SIZEOFFLOAT * num;
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
      size_t n = 1;
      ss << "dims[";
      for(int r = 0; r != rank-1; ++r) {
        ss << dims[r] << ",";
        n *= dims[r];
      }
      ss << dims[rank-1] << "]" << std::endl;
      n *= dims[rank-1];
      for(int i = 0; i != n; ++i) {
        ss << data[i] << " "; 
      } 
      ss << std::endl;
    };

    // return the tensor creation functions
    return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt};
  }
};



#pragma once

#include "../../src/tensor/tensor.h"
#include "../../src/ud_functions/ud_function.h"

#include <stdexcept>

using namespace bbts;
using tensor_args_t = ud_impl_t::tensor_args_t;
using ud_impl_callable = std::function<void(const bbts::ud_impl_t::tensor_params_t &params,
                                            const tensor_args_t &_in,
                                            tensor_args_t &_out)>;

#define PRINTLINE std::cout << __FILE__ << ": " << __LINE__ << std::endl

#define MAXRANK   4

// #define CU_BARB_USE_GPU
#define CU_BARB_USE_CPU

// just assume we're using CUDA_R_32F everywhere
#define SIZEOFFLOAT 4
cudaDataType_t cutensor_scalar_type = CUDA_R_32F;
cutensorComputeType_t cutensor_compute_type = CUTENSOR_COMPUTE_32F;

void handle_error(std::string msg, cutensorStatus_t status) {
  if(status == CUTENSOR_STATUS_SUCCESS){
    return;
  }

  // TODO: violently die
  std::cout << "VIOLENTLY DIEING{" << msg << "}" << std::endl;
  std::cout << "REASON: ";
  switch(status) {
    case CUTENSOR_STATUS_SUCCESS:                                                       break;
    case CUTENSOR_STATUS_NOT_INITIALIZED:        std::cout << "NOT_INITIALIZED"       ; break;
    case CUTENSOR_STATUS_ALLOC_FAILED:           std::cout << "ALLOC_FAILED"          ; break;
    case CUTENSOR_STATUS_INVALID_VALUE:          std::cout << "INVALID_VALUE"         ; break;
    case CUTENSOR_STATUS_ARCH_MISMATCH:          std::cout << "ARCH_MISMATCH"         ; break;
    case CUTENSOR_STATUS_MAPPING_ERROR:          std::cout << "MAPPING_ERROR"         ; break;
    case CUTENSOR_STATUS_EXECUTION_FAILED:       std::cout << "EXECUTION_FAILED"      ; break;
    case CUTENSOR_STATUS_INTERNAL_ERROR:         std::cout << "INTERNAL_ERROR"        ; break;
    case CUTENSOR_STATUS_NOT_SUPPORTED:          std::cout << "NOT_SUPPORTED"         ; break;
    case CUTENSOR_STATUS_LICENSE_ERROR:          std::cout << "LICENSE_ERROR"         ; break;
    case CUTENSOR_STATUS_CUBLAS_ERROR:           std::cout << "CUBLAS_ERROR"          ; break;
    case CUTENSOR_STATUS_CUDA_ERROR:             std::cout << "CUDA_ERROR"            ; break;
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE: std::cout << "INSUFFICIENT_WORKSPACE"; break;
    case CUTENSOR_STATUS_INSUFFICIENT_DRIVER:    std::cout << "INSUFFICIENT_DRIVER"   ; break;
    case CUTENSOR_STATUS_IO_ERROR:               std::cout << "IO_ERROR"              ; break;
  }
  std::cout << std::endl;
  throw std::runtime_error("cutensor didn't like something");
}
void handle_error(cutensorStatus_t status) {
  return handle_error("", status);
}

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

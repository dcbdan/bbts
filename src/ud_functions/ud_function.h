#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "../commands/command_utils.h"
#include "../server/static_config.h"
#include "../tensor/tensor.h"

#ifdef ENABLE_GPU
#include "../../third_party/cuda/gpu.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace bbts {

// the identifier of the ud function
// this should be used whenever we have something that maps or identifies the ud
// function
using ud_id_t = int32_t;

// uniquely identifies the implementation
using ud_impl_local_t = int32_t;
struct ud_impl_id_t {
  // the ud function identifier
  ud_id_t ud_id;

  // identifies the implementation withing the ud function
  ud_impl_local_t impl_id;
};

struct ud_impl_t {
  // the ud function arguments for either the input or the output
  template <class T>
  struct ud_impl_args_t {
    // just an empty argument list
    ud_impl_args_t() = default;

    // the input arguments
    ud_impl_args_t(std::vector<T *> in_arg) : arguments(std::move(in_arg)) {}

    // copy the input arguments
    ud_impl_args_t(std::vector<T> &in_arg) {
      // fill up the arguments
      arguments.reserve(in_arg.size());
      for (auto &p : in_arg) {
        arguments.push_back(&p);
      }
    }

    // just with empty arguments
    ud_impl_args_t(size_t num_args) : arguments(num_args) {}

    // the number of parameters
    size_t num_args() const { return arguments.size(); }

    // this is for the output arguments
    template <size_t n>
    T &get() {
      return *arguments[n];
    }

    // this is for the input arguments as they are constant
    template <size_t n>
    const T &get() const {
      return *arguments[n];
    }

    // this gets the arguments at runtime
    const T &get_by_idx(size_t idx) const { return *arguments[idx]; }

    // this gets the arguments at runtime
    T &get_by_idx(size_t idx) { return *arguments[idx]; }

    // sets the argument
    template <size_t n>
    void set(T &_in) {
      arguments[n] = &_in;
    }

    // set the argument on index
    void set(size_t n, T &_in) { arguments[n] = &_in; }

    void reinit(std::vector<T> &in_arg) {
      // fill up the arguments
      arguments.resize(in_arg.size());
      for (int32_t idx = 0; idx < in_arg.size(); ++idx) {
        arguments[idx] = &in_arg[idx];
      }
    }

    // resize the ud function
    void resize(size_t size) { arguments.resize(size); }

   private:
    // holds the input arguments
    std::vector<T *> arguments;
  };

  // define the arguments for the meta
  using tensor_args_t = ud_impl_args_t<tensor_t>;
  using meta_args_t = ud_impl_args_t<tensor_meta_t>;

  // define the parameters
  struct tensor_params_t {
    // returns the
    template <size_t n>
    float get_float() const {
      return _params[n].f;
    }

    template <size_t n>
    int32_t get_int() const {
      return _params[n].i;
    }

    template <size_t n>
    uint32_t get_uint() const {
      return _params[n].u;
    }

    template <size_t n>
    bool get_bool() const {
      return _params[n].b;
    }

    template <size_t n>
    float get_float_or_default(float val) const {
      if (n < _params.size()) {
        return _params[n].f;
      } else {
        return val;
      }
    }

    template <size_t n>
    int32_t get_int_or_default(int32_t val) const {
      if (n < _params.size()) {
        return _params[n].i;
      } else {
        return val;
      }
    }

    template <size_t n>
    uint32_t get_uint_or_default(uint32_t val) const {
      if (n < _params.size()) {
        return _params[n].u;
      } else {
        return val;
      }
    }

    template <size_t n>
    uint32_t get_bool_or_default(bool val) const {
      if (n < _params.size()) {
        return _params[n].b;
      } else {
        return val;
      }
    }

    command_param_t get_raw(size_t n) const { return _params[n]; }

    // returns the number of parameters
    size_t num_parameters() const { return _params.size(); }

    // the parameters
    bbts::command_param_list_t _params;

#ifdef ENABLE_GPU
    // the stream to use by the ud function
    cudaStream_t stream;

    // the handle to cublas
    cublasHandle_t cublas_handle;
#endif
  };

  // each apply is a call to these
  using ud_impl_callable =
      std::function<void(const bbts::ud_impl_t::tensor_params_t &params,
                         const tensor_args_t &_in, tensor_args_t &_out)>;

  // the impl_id of the implementation, this is initialized by the udf manager
  ud_impl_id_t impl_id;

  // the implementation name of this function this has to be unique
  // for example this could be mkl_matrix_multiplication or
  // stressen_matrix_multiplication
  std::string impl_name;

  // the name of the ud function, this is the same for all the ud functions that
  // create an equivalent result for example matrix_multiplication,
  // matrix_addition, etc..
  std::string ud_name;

  // the input types of the tensors
  std::vector<std::string> inputTypes;

  // tells the system what inputs can be inplace
  // an input can only be inplace if and only if there is one output
  // and that output can be the same tensor that input without chaning the
  // result
  std::vector<int32_t> inputInplace;

  // the output types
  std::vector<std::string> outputTypes;

  // does this implementation require the tensors to be on the gpu
  bool is_gpu;

  // make the virtual destructor
  virtual ~ud_impl_t() = default;

  // this calls the function, it will do different things depending on whether
  // the ud function is using the GPU or not
  void call_ud(const bbts::ud_impl_t::tensor_params_t &_params,
               const tensor_args_t &_in, tensor_args_t &_out) const;

  // call the gpu kernel if this is a gpu function
  void call_gpu_ud(const bbts::ud_impl_t::tensor_params_t &_params,
                   const tensor_args_t &_in, tensor_args_t &_out) const;

  // this is a function pointer to the function that needs to be applied
  // we don't use virtual function on purpose applied function can be a free
  // standing function
  ud_impl_callable fn;

  // the gpu function to call
  ud_impl_callable gpu_fn;

  // returns the complexity hint of the ud function
  virtual size_t get_complexity_hint(
      const bbts::ud_impl_t::tensor_params_t &params,
      const meta_args_t &_in) = 0;

  // returns the output meta data
  virtual void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                            const meta_args_t &_in,
                            meta_args_t &_out) const = 0;
};

// define a nice way to say unique_ptr of ud_impl_t
using ud_impl_ptr_t = std::unique_ptr<ud_impl_t>;

struct ud_func_t {
  // the impl_id of the implementation, this is initialized by the udf manager
  ud_id_t id;

  // the name of the ud function, this is the same for all the ud functions that
  // create an equivalent result for example matrix_multiplication,
  // matrix_addition, etc..
  std::string ud_name;

  // this tells us whether the ud function is associative, or not
  bool is_ass;

  // this tells us whether the ud function commutative, or not
  bool is_com;

  // this tells us how many tensors we need as an input to this function
  size_t num_in;

  // this tells us how tensors this function is outputting
  size_t num_out;

  // these are all the implementations of this ud function
  std::vector<ud_impl_ptr_t> impls;

  // adds the implementation to the ud function
  ud_impl_id_t add_impl(ud_impl_ptr_t _impl) {
    // add the implementation to the list of all implementations
    auto impl_id = static_cast<ud_impl_local_t>(impls.size());
    _impl->impl_id = ud_impl_id_t{.ud_id = id, .impl_id = impl_id};
    impls.emplace_back(std::move(_impl));

    return impls.back()->impl_id;
  }
};

// define a nicer way to say unique_ptr of ud_func_t
using ud_func_ptr_t = std::unique_ptr<ud_func_t>;

}  // namespace bbts

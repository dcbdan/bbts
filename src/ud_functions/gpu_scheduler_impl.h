#pragma once

#include <future>
#include <condition_variable>
#include <mutex>
#include <queue>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cutensor.h>
#include "../tensor/builtin_formats.h"
#include "../ud_functions/builtin_functions.h"
#include "../tensor/tensor_factory.h"
#include "../../third_party/cuda/gpu.h"

namespace bbts {

struct gpu_scheduler_impl_t {

  // 
  gpu_scheduler_impl_t(const bbts::tensor_factory_ptr_t &fact);

  // free everything
  ~gpu_scheduler_impl_t();

  // run the scheduler
  void run();

  // runs the kernel
  std::future<bool> execute_kernel(bbts::ud_impl_t* fun,
                                   const bbts::ud_impl_t::tensor_params_t* params,
                                   const bbts::ud_impl_t::tensor_args_t* inputs,
                                   bbts::ud_impl_t::tensor_args_t* outputs);

  // shutdown the the scheduler
  void shutdown();

private:

  // rotate everything
  void _rotate();

  // specifies the kernel function to run and the parameters
  struct kernel_spec_t {

    // the kernel we want to run
    bbts::ud_impl_t* fun;

    // the input parameters
    bbts::ud_impl_t::tensor_params_t params;

    // the inputs
    const bbts::ud_impl_t::tensor_args_t* inputs;

    // the outputs
    bbts::ud_impl_t::tensor_args_t* outputs;

    // did succeed running it?
    std::promise<bool> success;
  };
  
  // used to sync 
  std::mutex _m;
  std::condition_variable _cv;

  // the queue of things to process
  std::queue<kernel_spec_t> _q;

  // the streams FRONT, MID, BACK
  cudaStream_t _streams[3];
  cudaEvent_t  _events[3];

  // the handle
  cublasHandle_t   _cub_handles[3];
  cutensorHandle_t _cut_handle;

  // do we have something to do
  std::uint32_t _left_to_process = 0;

  // is it shutdown
  bool _shutdown = false;

  // do we have something to do on the FRONT, MID, BACK stream
  bool _has_something[3] = {false, false, false};

  // the kernels we are running  
  kernel_spec_t _specs[3] = {0};

  // the tensor factory
  bbts::tensor_factory_ptr_t _factory;

  // the stream
  size_t FRONT = 0;
  size_t MID = 1;
  size_t BACK = 2;
};

}

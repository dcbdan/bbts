#include "gpu_scheduler_impl.h"

namespace bbts {

gpu_scheduler_impl_t::gpu_scheduler_impl_t(const bbts::tensor_factory_ptr_t &fact) : _factory(fact) {

    // create the front stream (at the current step fetch the tensor)
    cudaStreamCreate(&_streams[FRONT]);
    cudaEventCreate(&_events[FRONT]);
    cublasCreate(&_cub_handles[FRONT]); cublasSetStream(_cub_handles[FRONT], _streams[FRONT]);

    // create the mid stream (at the current step do the kernel)
    cudaStreamCreate(&_streams[MID]);
    cudaEventCreate(&_events[MID]);
    cublasCreate(&_cub_handles[MID]); cublasSetStream(_cub_handles[MID], _streams[MID]);
    cutensorInit(&_cut_handle);

    // create the back stream (at the current step does unloading)
    cudaStreamCreate(&_streams[BACK]);
    cudaEventCreate(&_events[BACK]);
    cublasCreate(&_cub_handles[BACK]); cublasSetStream(_cub_handles[BACK], _streams[BACK]);

}


gpu_scheduler_impl_t::~gpu_scheduler_impl_t() {

    // destroy the cublas handles
    cublasDestroy(_cub_handles[FRONT]);
    cublasDestroy(_cub_handles[MID]);
    cublasDestroy(_cub_handles[BACK]);
}

void gpu_scheduler_impl_t::run() {

  while (true) {
  
    // sync the phases
    cudaEventSynchronize(_events[FRONT]);
    cudaEventSynchronize(_events[MID]);
    cudaEventSynchronize(_events[BACK]);

    // wait until we have something here
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk, [&]{ return _left_to_process != 0 || !_q.empty() || _shutdown; });
    
    // are we done?
    if(_shutdown && _left_to_process == 0) {
      break;
    }

    // run the kernel first, since this is more imporant
    if(_has_something[MID]) {

      // set the stream and cublas handle
      _specs[MID].params.stream = _streams[MID]; 
      _specs[MID].params.cublas_handle = _cub_handles[MID];
      _specs[MID].params.cutensor_handle = _cut_handle;

      // call the kernel
      _specs[MID].fun->call_gpu_ud( _specs[MID].params, *_specs[MID].inputs, *_specs[MID].outputs);
      cudaEventRecord(_events[MID], _streams[MID]);
    }

    // do we have something to put into the pipeline
    if(!_q.empty()) {

      // get it from the queue
      _specs[FRONT] = std::move(_q.front()); _q.pop();

      // prefetch the input tensors
      for(std::size_t i = 0; i < _specs[FRONT].inputs->num_args(); ++i) {

        // get the tensor and the number of bytes
        auto &ts = _specs[FRONT].inputs->get_by_idx(i)._blob;
        auto num_bytes = _factory->get_tensor_size(_specs[FRONT].inputs->get_by_idx(i)._meta) - sizeof(bbts::tensor_meta_t);
        cudaMemPrefetchAsync(&ts, num_bytes, 0, _streams[FRONT]);
      }
      cudaEventRecord(_events[FRONT], _streams[FRONT]);
      _has_something[FRONT] = true;

      // we have one more to process
      _left_to_process++;
    }

    // do the unloading if necessary
    if(_has_something[BACK]) {

      // go through all the outputs and unload
      for(std::size_t i = 0; i < _specs[BACK].outputs->num_args(); ++i) {

        // get the tensor and the number of bytes
        auto &ts = _specs[BACK].outputs->get_by_idx(i)._blob;
        auto num_bytes = _factory->get_tensor_size(_specs[BACK].outputs->get_by_idx(i)._meta) - sizeof(bbts::tensor_meta_t);

        // load out on back
        cudaMemPrefetchAsync(&ts, num_bytes, cudaCpuDeviceId, _streams[BACK]); 
      }
      cudaEventRecord(_events[BACK], _streams[BACK]); 
      _has_something[BACK] = false;

      // we finished
      _specs[BACK].success.set_value(true);
      
      // we processed one
      _left_to_process--;
    }

    // rotate the streams and events
    _rotate();
  }
}

std::future<bool> gpu_scheduler_impl_t::execute_kernel(bbts::ud_impl_t* fun,
                                                  const bbts::ud_impl_t::tensor_params_t * params,
                                                  const bbts::ud_impl_t::tensor_args_t* inputs,
                                                  bbts::ud_impl_t::tensor_args_t* outputs) {

  std::future<bool> wait;
  {
    std::promise<bool> success;
    wait = success.get_future();

    // lock this thing
    std::unique_lock<std::mutex> lk(_m);
    
    // schedule
    _q.push(kernel_spec_t{.fun = fun, .params = *params, .inputs = inputs, .outputs = outputs, .success = std::move(success)});
    _cv.notify_one();
  }

  return wait;
}

void gpu_scheduler_impl_t::shutdown() {

  std::unique_lock<std::mutex> lk(_m);

  // shutdown and notify
  _shutdown = true;
  _cv.notify_one();
}

void gpu_scheduler_impl_t::_rotate() {

  auto tmp = BACK;
  BACK  = MID;
  MID   = FRONT;
  FRONT = tmp;
}


}

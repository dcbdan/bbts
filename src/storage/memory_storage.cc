#include "memory_storage.h"

#ifdef ENABLE_GPU
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#endif

#include "../server/static_config.h"
#include "../utils/terminal_color.h"
#include <iostream>
#include <sstream>


namespace bbts {

memory_storage_t::~memory_storage_t() {

  // go through each allocated tensor and free it
  for(auto &it : _tensor_nfo) {
    free_tensor(it.second.address);
  }
}

memory_storage_t::tensor_ref_t memory_storage_t::_get_by_tid(tid_t _id) {

  // try to find the tensor if we find it return the address
  auto it = _tensor_nfo.find(_id);
  return it != _tensor_nfo.end() ? tensor_ref_t{ .id = _id, .tensor = it->second.address } :
                                   tensor_ref_t{ .id = _id, .tensor = nullptr };
}

memory_storage_t::tensor_ref_t memory_storage_t::_create_tensor(tid_t _id, size_t num_bytes) {

  // malloc the tensor
  tensor_t *ts = _allocate_tensor(num_bytes);

  // store the info
  _tensor_nfo[_id] = sto_tensor_nfo_t{.address = ts, .num_bytes = num_bytes};

  // notify that the tensor is created
  if constexpr (static_config::enable_hooks) {
    _tensor_create_hook(_id);
  }

  // return the tensor
  return {.id = _id, .tensor = ts};
}

memory_storage_t::tensor_ref_t memory_storage_t::_create_tensor(size_t num_bytes) {

  // malloc the tensor
  tensor_t *ts = _allocate_tensor(num_bytes);

  // get a new tid for this
  auto tid = _current_anon--;
  _tensor_nfo[tid] = { .address=ts, .num_bytes=num_bytes};

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // notify that the tensor is created
    _tensor_create_hook(TID_NONE);
  }

  // return the tensor
  return {.id = tid, .tensor = ts};
}

tensor_t *memory_storage_t::_allocate_tensor(size_t num_bytes) {

  // malloc the tensor
  tensor_t *ts;

  // check if we even support the GPU
  if constexpr(static_config::enable_gpu) {
    #ifdef ENABLE_GPU
    // allocate the GPU
    checkCudaErrors(cudaMallocManaged(&ts, num_bytes));
    #endif
  }
  else {

    // we can not do this
    ts = (tensor_t*) malloc(num_bytes);
  }

  return ts;
}

void memory_storage_t::free_tensor(tensor_t *tensor) {

  // check if we even support the GPU
  if constexpr(static_config::enable_gpu) {
    #ifdef ENABLE_GPU
    // free the GPU
    checkCudaErrors(cudaFree(tensor));
    #endif
  }
  else {

    // free the regular tensor
    free(tensor);
  }
}

bool memory_storage_t::has_tensor(tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // check if found
  return _tensor_nfo.find(_id) != _tensor_nfo.end();
}

bool memory_storage_t::remove_by_tid(tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // remove the tensor
  auto it = _tensor_nfo.find(_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // free the tensor
  free_tensor(it->second.address);

  // remove the tensor
  _tensor_nfo.erase(it);

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // call that the tensor is deleted
    _tensor_delete_hook(_id);
  }

  return true;
}

bool memory_storage_t::assign_tid(tid_t _anon_id, tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find it
  auto it = _tensor_nfo.find(_anon_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // rewire
  _tensor_nfo[_id] = it->second;
  _tensor_nfo.erase(it);

  return true;
}

size_t memory_storage_t::get_num_tensors() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  return _tensor_nfo.size();
}

size_t memory_storage_t::get_tensor_size(tid_t _id){

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // return the size if the tensor exists
  auto it = _tensor_nfo.find(_id);
  return it == _tensor_nfo.end() ? 0 : it->second.num_bytes;
}

void memory_storage_t::print(std::stringstream &ss) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // print all the allocated tensors
  ss << bbts::green << "TID\tSize (in bytes)\t\taddress\n" << bbts::reset;
  for(auto &t : _tensor_nfo) {
    ss << t.first << "\t" << t.second.num_bytes << "\t\t" << (void*) t.second.address << '\n';
  }
}

void memory_storage_t::clear() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // go through each allocated tensor and free it
  for(auto &it : _tensor_nfo) {

    // is it gpu
    free_tensor(it.second.address);
  }
  _tensor_nfo.clear();
}

memory_storage_t::reservation_result_t
memory_storage_t::_create_reserved(
  const std::vector<tid_t> &get,
  const std::vector<std::tuple<tid_t, size_t>> &create_or_get)
{
  // get all the tensors
  std::vector<tensor_ref_t> out_get;
  out_get.reserve(get.size());
  for (auto t : get) {
    out_get.push_back(_get_by_tid(t));
  }

  // create all the tensors we need
  std::vector<tensor_ref_t> out_create;
  out_create.reserve(create_or_get.size());
  for (auto [id, num_bytes] : create_or_get) {
    // create (or get) the tensor.
    // The three cases are
    // 1. the tid is not valid and the tensor has not been created
    // 2. the tid is     valid and the tensor has     been created
    // 3. the tid is     valid and the tensor has not been created
    if (id == TID_NONE) {
      out_create.push_back(_create_tensor(num_bytes));
    } else {
      // we have a valid id, but has the tensor been created?
      auto tensor_ref_maybe = _get_by_tid(id);
      if(tensor_ref_maybe.tensor) {
        // the tensor address is valid so it must have been created
        out_create.push_back(tensor_ref_maybe);
      } else {
        // the tensor has not been created, so create it
        out_create.push_back(_create_tensor(id, num_bytes));
      }
    }
  }

  // return the result
  return {.get = out_get, .create_or_get = out_create};
}

std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> memory_storage_t::extract_meta() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  std::size_t idx = 0;
  std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> out(_tensor_nfo.size());
  for(auto nfo : _tensor_nfo) {
    out[idx++] = {nfo.first, nfo.second.address->_meta};
  }

  return std::move(out);
}


}

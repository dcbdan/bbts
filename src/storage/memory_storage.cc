#include "memory_storage.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "../communication/ib_communicator.h"
#include "../communication/infiniband/mr_bytes.h"
#include "../server/static_config.h"
#include "../utils/terminal_color.h"
#include <iostream>
#include <sstream>
#include <sys/mman.h>
//#include <jemalloc/jemalloc.h>

namespace bbts {

memory_storage_t::memory_storage_t(communicator_ptr_t com,
                                   size_t to_allocate)
{

  // just empty hooks
  _tensor_create_hook = [](tid_t _) {};
  _tensor_delete_hook = [](tid_t _) {};

  // get ourselves a memory region
    _tensor_memory = com->create_and_use_tensor_memory_region(to_allocate);
  // and just copy the address to _mem
  _mem = (uint8_t*)_tensor_memory->get_bytes().data;

  // make the block allocator
  _allocator = std::make_shared<block_allocator_t>(to_allocate);
}

memory_storage_t::~memory_storage_t() {
  // go through each allocated tensor and free it
  for(auto &it : _tensor_nfo) {
    free_tensor(it.second.address, it.second.num_bytes);
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

  // allocate the host pinned memory
  auto offset = _allocator->allocate(num_bytes);
  ts = (tensor_t*) (_mem + offset);

  return ts;
}

void memory_storage_t::free_tensor(tensor_t *tensor, size_t num_bytes) {
  // free the memory
  size_t offset = (size_t)((uint8_t*) tensor - _mem);
  _allocator->free(offset, num_bytes);
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
  free_tensor(it->second.address, it->second.num_bytes);

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
    free_tensor(it.second.address, it.second.num_bytes);
  }
  _tensor_nfo.clear();
}

memory_storage_t::reservation_result_t memory_storage_t::_create_reserved(const std::vector<tid_t> &get,
                                                                         const std::vector<std::tuple<tid_t, size_t>> &create) {

  // get all the tensors
  std::vector<tensor_ref_t> out_get;
  out_get.reserve(get.size());
  for (auto t : get) {
    out_get.push_back(_get_by_tid(t));
  }

  // create all the tensors we need
  std::vector<tensor_ref_t> out_create;
  out_create.reserve(create.size());
  for (auto ct : create) {

    // create all the necessary tensors
    auto [id, num_bytes] = ct;
    if (id != TID_NONE) {
      out_create.push_back(_create_tensor(id, num_bytes));
    } else {
      out_create.push_back(_create_tensor(num_bytes));
    }
  }

  // return the result
  return {.get = out_get, .create = out_create};
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

} // namespace bbts

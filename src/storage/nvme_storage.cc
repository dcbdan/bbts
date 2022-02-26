#include "nvme_storage.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>

#ifdef ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <cuda_runtime_api.h>
#endif

#include "../server/static_config.h"
#include "../utils/terminal_color.h"
#include <iostream>
#include <future>
#include <sstream>
#include <stdexcept>
#include <unistd.h>


namespace bbts {

nvme_storage_t::~nvme_storage_t() {

  // free all the tensors
  for(auto &nfo : _tensor_nfo) {
    if(nfo.second.state == tensor_state_t::LOADED) {
      free_tensor(nfo.second.data.get().tensor);
    }
  } 
}

void nvme_storage_t::request_thread() {

  while (true) {

    // lock this thing
    std::unique_lock<std::mutex> lck (_m);

    // wait until we have something
    _res_processing_cv.wait(lck, [&]{return !_scheduled_reservations.empty() || _is_shutdown;});

    // break out from here if we are done
    if(_is_shutdown) {
      break;
    }

    // get the reservation
    auto val = std::move(std::get<0>(_scheduled_reservations.front()));
    auto res = std::get<1>(_scheduled_reservations.front());

    // pop it
    _scheduled_reservations.pop();

    // the result
    reservation_result_t out(res.get->size(), res.create->size());

    // we load the tensors here
    std::vector<std::tuple<sto_tensor_nfo_t*, tensor_t*>> to_load;
    to_load.reserve(res.get->size());

    // go through all the tensor we need to get
    size_t required = 0;
    for(auto &g : *res.get) {
      
      // find the tensor
      auto it = _tensor_nfo.find(g);
      
      // make sure the tensor is not deleted
      assert(it->second.state != tensor_state_t::DELETED);

      // if the tensor is not loaded we need prep, if it is unloading 
      // we will take care of it later but just not freeing the memory
      if(it->second.state == tensor_state_t::NOT_LOADED) {

        // make a new promise
        it->second.promise = std::promise<tensor_ref_t>();

        // turn in into a new shared future
        it->second.data = std::shared_future<tensor_ref_t>(it->second.promise.get_future());
        
        // mark that we are the one loading this tensor
        it->second.state = tensor_state_t::LOADING;

        // mark that we are loading this
        to_load.push_back({&it->second, nullptr});

        // increment the number of bytes reqired
        required += it->second.num_bytes; 
      }

      // if the tensor is loaded 

      // store the future
      // this is valid for UNLOADING, LOADING, LOADED
      out.get.push_back(it->second.data);
    }
    
    // go through all the tensors we need to create
    for(auto &c : *res.create) {
      
      // the the info about the tensor we need to create
      auto [tid, num_bytes] = c;

      // make an entry
      auto &ts = _tensor_nfo[tid];

      // set the values
      ts.id = tid;
      ts.num_bytes = num_bytes;
      ts.num_ref = 1;
      ts.file_offset = -1;
      ts.state = tensor_state_t::LOADED;
      ts.promise = std::promise<tensor_ref_t>();
      ts.data = std::shared_future<tensor_ref_t>(ts.promise.get_future());

      // we need to allocate this
      required += num_bytes;

      // add the create
      out.create.push_back(ts.data);
    }

    // evict some tensors if necessary
    if(_cur_allocated + required >= _max_allocated) {
      _evict_some(lck, required);
    }

    // allocate the necessary tensors
    for(auto &c : *res.create) {
      
      auto [tid, num_bytes] = c;

      // get an entry
      auto &ts = _tensor_nfo[tid];

      // set the value
      ts.promise.set_value(tensor_ref_t{.id = tid, .tensor = _allocate_tensor(num_bytes)});

      // we just allocated
      _cur_allocated += num_bytes;
    }

    // load all the tensors
    for(auto &l : to_load) {
      
      // get the info and where we want to allocate the tensor
      auto &[nfo, t] = l;
    
      // allocate the tensor
      t = _allocate_tensor(nfo->num_bytes);

      // we just allocated
      _cur_allocated += nfo->num_bytes;
    }

    for(auto &l : to_load) {

      // get the info and where we want to allocate the tensor
      auto &[nfo, t] = l;

      auto num_bytes = nfo->num_bytes;
      auto offset = nfo->file_offset;

      // read the tensor
      lck.unlock();
      pread(_fd, t, num_bytes, offset);
      lck.lock();

      // allocate the tensor
      nfo->state = tensor_state_t::LOADED;
      nfo->promise.set_value(tensor_ref_t{.id = nfo->id, .tensor = t});
    }

    // set the value
    val.set_value(out);
  }

}

bool nvme_storage_t::remove_by_tid(tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find it
  auto it =_tensor_nfo.find(_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // make sure that nobody is referencing this tensor
  assert(it->second.num_ref == 0);

  // check for special cases
  if(it->second.state == tensor_state_t::UNLOADING) {
    
    // if the tensor is unloading now just mark it as deleted and return
    it->second.state = tensor_state_t::DELETED;
    return true;
  } 
  else if(it->second.state == tensor_state_t::NOT_LOADED) {
    
    // if it is not laded, simpy removce it
    _tensor_nfo.erase(it);
    return true;
  }

  // remove it from the lru
  _lru.remove(_id);

  // it is no longer allocated 
  _cur_allocated -= it->second.num_bytes;

  // free the memory allocated
  free_tensor(it->second.data.get().tensor);

  // remove it
  _tensor_nfo.erase(it);

  return true;
}

bool nvme_storage_t::assign_tid(tid_t _anon_id, tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find it
  auto it =_tensor_nfo.find(_anon_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // make sure that nobody is referencing this tensor
  assert(it->second.num_ref == 0);

  // reassign
  auto &nfo = _tensor_nfo[_id];
  nfo.data = it->second.data;
  nfo.file_offset = it->second.file_offset;
  nfo.id = it->second.id;
  nfo.num_bytes = it->second.num_bytes;
  nfo.num_ref = it->second.num_ref;
  nfo.state = it->second.state;
  nfo.promise = std::move(it->second.promise);
  nfo.data = it->second.data;

  // check for special cases
  if(it->second.state == tensor_state_t::UNLOADING) {

    // if the tensor is unloading now just mark it as deleted and return
    it->second.state = tensor_state_t::REASSIGNED;
    it->second.id = _id;
  }
  else {

    // remove it from the lru
    _lru.reassign(_anon_id, _id);

    // remove it
    _tensor_nfo.erase(it);
  }

  return true;
}

void nvme_storage_t::set_max_storage(size_t val) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // check if this is smaller, dump some to disk
  if(val < _max_allocated) {
    _max_allocated = val;
    _evict_some(lck, _max_allocated - val);
  }
}

size_t nvme_storage_t::get_num_tensors() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // return the number of tensors in the system
  return _tensor_nfo.size();
}

size_t nvme_storage_t::get_tensor_size(tid_t id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find it
  auto it = _tensor_nfo.find(id);
  if(it == _tensor_nfo.end()) {
    return 0;
  }

  // return the tensor size
  return it->second.num_bytes;
}

void nvme_storage_t::print(std::stringstream &ss) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // print all the allocated tensors
  ss << bbts::green << "TID\tSize (in bytes)\t\taddress\n" << bbts::reset;
  for(auto &t : _tensor_nfo) {

    // get the address
    void *address = t.second.state == tensor_state_t::LOADED ? t.second.data.get().tensor : nullptr;

    // print it out
    ss << t.first << "\t" << t.second.num_bytes << "\t\t" << (void*) address << '\n';
  }
}

void nvme_storage_t::clear() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // free all the tensors
  for(auto &nfo : _tensor_nfo) {
    if(nfo.second.state == tensor_state_t::LOADED) {
      free_tensor(nfo.second.data.get().tensor);
    }
  }

  // clear the nfo
  _tensor_nfo.clear();

  // set all the offsets to zero
  _file_offset = 0;
  _cur_reserved = 0;
  _cur_allocated = 0;
  _current_anon = TID_NONE - 1;
  _lru.clear();
  ftruncate(_fd, 0);
}

void nvme_storage_t::shutdown() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // shutdown the storage
  _is_shutdown = true;
  _res_processing_cv.notify_all();
}

bool nvme_storage_t::_try_reserve(const std::vector<tid_t> &get,
                                  std::vector<std::tuple<tid_t, size_t>> &create) {

  // go through all the tensor we need to get
  size_t required = 0;
  for(auto &t : get) {

    // check if anybody has already reserved this tensor
    // if so there is no need to budget memory for it as it was already done
    auto it = _tensor_nfo.find(t);
    if(it->second.num_ref == 0) {
      required += it->second.num_bytes;
    } 
  }

  // go through all the tensors we want to create
  for(auto &c : create) {

    // add it to the required
    required += std::get<1>(c);
  }

  // make sure it is acutally possible to process this
  if(required >= _max_allocated) {
    throw std::runtime_error("Can not process this.");
  }

  // check if we have enough memory
  if(_cur_reserved + required >= _max_allocated) {

    // we failed
    return false;
  }

  // we need to reverse reference counts
  for(auto &t : get) {
    // remove it from the lru if it is there and increment the reference count
    auto it = _tensor_nfo.find(t);
    _lru.remove(it->second.id);
    it->second.num_ref++;
  }

  // assign a tid to the anonymous tensor
  for(auto &c : create) {
    if(std::get<0>(c) == TID_NONE) {
      std::get<0>(c) = _current_anon--;
    }
  }

  // mark this as reserved
  _cur_reserved += required;

  // tell that we have approved it
  return true;
}

bool nvme_storage_t::has_tensor(tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // find information about the tensor
  auto it = _tensor_nfo.find(_id);

  // check if we found it
  return it != _tensor_nfo.end();
}

std::future<nvme_storage_t::reservation_result_t> nvme_storage_t::_create_reserved(const std::vector<tid_t> &get,
                                                                                   const std::vector<std::tuple<tid_t, size_t>> &create) {
  

  // get the future that is going to return the reservation result
  std::promise<reservation_result_t> res;
  auto f = res.get_future();

  // add the reservation to the queue
  _scheduled_reservations.push({std::move(res), reservation_nfo_t{.get = &get, .create = &create}});

  // make sure the processing threads know that there is stuff to do
  _res_processing_cv.notify_all();

  // return the result
  return std::move(f);
}

void nvme_storage_t::_release_reservation(const std::vector<tid_t> &get,
                                          const std::vector<std::tuple<tid_t, size_t>> &create) {
  
  // go through all the tensor we wanted to get
  for(auto &t : get) {

    // find information about the tensor
    auto it = _tensor_nfo.find(t);
    it->second.num_ref--;

    // if this tensor is no longer reserved mark that we are 
    if(it->second.num_ref == 0) {

      // ok this is not reserved anymore
      _cur_reserved -= it->second.num_bytes;

      // if we already laoded it add it to eviction
      if(it->second.state == tensor_state_t::LOADED) {
        
        // add it to the lru 
        _lru.add(t);
      }
    }
  }

  // go through all the tensors we wanted to create
  for(auto &c : create) {

    // find the created tensor
    auto it = _tensor_nfo.find(std::get<0>(c));
    it->second.num_ref--;

    // should be zero as otherwise it is wrong usage
    assert(it->second.num_ref == 0);

    // remove the reserved
    _cur_reserved -= std::get<1>(c);

    // add it to the lru, as it should have zero references
    _lru.add(it->second.id);
  }
}

void nvme_storage_t::_cancel_reservation(const std::vector<tid_t> &get,
                                         const std::vector<std::tuple<tid_t, size_t>> &create) {
  
  // go through all the tensor we wanted to get
  for(auto &t : get) {

    // find information about the tensor
    auto it = _tensor_nfo.find(t);
    it->second.num_ref--;

    // if this tensor is no longer reserved mark that we are 
    if(it->second.num_ref == 0) {

      // ok this is not reserved anymore
      _cur_reserved -= it->second.num_bytes;

      // if we already laoded it add it to eviction
      if(it->second.state == tensor_state_t::LOADED) {
        
        // add it to the lru 
        _lru.add(t);
      }
    }
  }

  // go through all the tensors we wanted to create
  for(auto &c : create) {
    
    // remove the reserved
    _cur_reserved -= std::get<1>(c);
  }

}

tensor_t *nvme_storage_t::_allocate_tensor(size_t num_bytes) {

  // malloc the tensor
  tensor_t *ts;
  if constexpr(static_config::enable_gpu) {
    #ifdef ENABLE_GPU
    // allocate the GPU
    checkCudaErrors(cudaMallocManaged(&ts, num_bytes));
    #endif
  }
  else {
    
    // this is a CPU
    ts = (tensor_t*) malloc(num_bytes); 
  }

  return ts;
}

void nvme_storage_t::free_tensor(tensor_t *tensor) {

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

void nvme_storage_t::_evict_some(std::unique_lock<std::mutex> &lck, size_t required) {
  
  while (_cur_allocated + required >= _max_allocated) {

    // evict a tensor
    auto id = _lru.evict(lck);

    // check if it is deleted
    auto it = _tensor_nfo.find(id);

    // just some checking to make sure we are cool to evict this
    assert(it->second.num_ref == 0);
    assert(it->second.state == tensor_state_t::LOADED);

    // change the state to unloading
    it->second.state = tensor_state_t::UNLOADING;

    // if it does not have an offset assigned give it one
    if(it->second.file_offset == -1) {
      it->second.file_offset = _file_offset;
      _file_offset += it->second.num_bytes;
    }
    
    // unlock so we can dump the data to the disk
    lck.unlock();

    pwrite(_fd, it->second.data.get().tensor, it->second.num_bytes, it->second.file_offset);

    // lock again so we can update the state
    lck.lock();

    // check if the tensor was resassigned
    if(it->second.state == tensor_state_t::REASSIGNED) {
      it = _tensor_nfo.find(it->second.id);
    }

    // check if somebody has used it in the mean time
    if(it->second.num_ref != 0) {

      // ok we can not release this 
      it->second.state = tensor_state_t::LOADED;
    }
    else if(it->second.state == tensor_state_t::DELETED) {
      
      // free the memory
      free_tensor(it->second.data.get().tensor);
      _cur_allocated -= it->second.num_bytes;

      // if the tensor was delted in the mean time just kill it and add the memory
      _tensor_nfo.erase(it);
    }
    else {

      // ok the tensor is still there and it is not used
      _cur_allocated -= it->second.num_bytes;
      it->second.state = tensor_state_t::NOT_LOADED;

      // free the memory
      free_tensor(it->second.data.get().tensor);
    }
  }
}

std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> nvme_storage_t::extract_meta() {

  std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> out;
  for(auto &t : _tensor_nfo) {

    // if the tensor was deleted or reassined 
    if(t.second.state == tensor_state_t::DELETED || t.second.state == tensor_state_t::REASSIGNED) {
      continue;
    }

    // run the transaction
    local_transaction({t.first}, {}, [&](const nvme_storage_t::reservation_result_t &res) {

      // the get the tensor
      auto ts = res.get[0].get().tensor;
      out.push_back({t.first, ts->_meta});
    });
  }

  return std::move(out);
}

}


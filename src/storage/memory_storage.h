#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <memory>
#include <mutex>

#include "block_allocator.h"

#include "../tensor/tensor.h"
#include "../communication/communicator.h"

namespace bbts {


// this class is responsible for memory managment like getting new
// memory for tensors, getting already existing tensors and in the future evicting
// tensors to disk or moving them to GPU
struct memory_storage_t {

  memory_storage_t(communicator_ptr_t com, size_t to_allocate);

  // destructor
  ~memory_storage_t();

  // is a reference of the tensor
  struct tensor_ref_t {

    // the identifier of the tensor
    tid_t id;

    // tensor
    tensor_t *tensor;

    // this is added to be complient with the interface of the nvme storage
    // where a shared_future is used for the tensor reference
    const tensor_ref_t &get() const { return *this; }
  };

  // the result of a reservation
  struct reservation_result_t {

    // the existing tensors we want to get
    std::vector<tensor_ref_t> get;

    // the tensors we want to create
    std::vector<tensor_ref_t> create;
  };


  // make sure that all the tensors created or requested are aquired at the same time
  template<class fn>
  void local_transaction(const std::vector<tid_t> &get,
                         const std::vector<std::tuple<tid_t, size_t>> &create,
                         const fn &fun) {

    // lock this thing
    std::unique_lock<std::mutex> lck (_m);

    // craete the reserved
    auto c = _create_reserved(get, create);

    // run the function
    lck.unlock();
    fun(c);
  }

  // if there is a
  template<class fn>
  void remote_transaction(command_id_t cmd,
                          const bbts::command_t::node_list_t &nodes,
                          const std::vector<tid_t> &get,
                          const std::vector<std::tuple<tid_t, size_t>> &create,
                          const fn &fun) {

    // lock this thing
    std::unique_lock<std::mutex> lck (_m);

    // craete the reserved
    auto c = _create_reserved(get, create);

    // run the function
    lck.unlock();
    fun(c);
  }

    // if there is a
  template<class fn>
  void remote_transaction_p2p(command_id_t cmd,
                              node_id_t other,
                              const std::vector<tid_t> &get,
                              const std::vector<std::tuple<tid_t, size_t>> &create,
                              const fn &fun) {

    // lock this thing
    std::unique_lock<std::mutex> lck (_m);

    // craete the reserved
    auto c = _create_reserved(get, create);

    // run the function
    lck.unlock();
    fun(c);
  }

  // allocate the tensor
  tensor_t *_allocate_tensor(size_t num_bytes);

  // free the allocated tensor
  void free_tensor(tensor_t *tensor, size_t num_bytes);

  // check if there is a tensor in the storage
  bool has_tensor(tid_t _id);

  // remove by tid
  bool remove_by_tid(tid_t _id);

  // assign a tid ot the anonymous tensor
  bool assign_tid(tid_t _anon_id, tid_t _id);

  // print the memory_storage
  void print(std::stringstream &ss);

  // clear the memory_storage
  void clear();

  // get the number of tensors in the system
  size_t get_num_tensors();

  // returns the size of a tensor
  size_t get_tensor_size(tid_t _id);

  // add the retired hook
  template<class fn>
  void add_created_hook(fn fun){ _tensor_create_hook = fun; }

  // add the retired hook
  template<class fn>
  void add_deleted_hook(fn fun){ _tensor_delete_hook = fun; }

  // just an empty request thread
  void request_thread() {};

  // shutdown
  void shutdown() {};

  // extract the meta
  std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> extract_meta();

private:

  // information about the stored tensor
  struct sto_tensor_nfo_t {

    // if present the address will be not null if evicted
    tensor_t *address;

    // the size of the tensor in bytes
    size_t num_bytes;
  };

  // returns a tensor for the tid, the tensor is always not initialized
  tensor_ref_t _create_tensor(tid_t _id, size_t num_bytes);

  // returns an anonymous tensor, the tensor is always not initialized
  tensor_ref_t _create_tensor(size_t num_bytes);

  // an existing tensor by tid
  tensor_ref_t _get_by_tid(tid_t _id);

  // craete all the tensors we just reserved
  reservation_result_t _create_reserved(const std::vector<tid_t> &get,
                                       const std::vector<std::tuple<tid_t, size_t>> &create);

  // the mutex to lock this thing as it is going to be hammered by threads
  std::mutex _m;

  // all the allocated host memory
  std::shared_ptr<ib::memory_region_bytes_t> _tensor_memory;
  // and a reference to the bytes contained within
  uint8_t* _mem;

  // this gives us the allocated memory block
  block_allocator_ptr_t _allocator;

  // init it to the first available negative number
  tid_t _current_anon = TID_NONE - 1;

  // maps to the information
  std::unordered_map<tid_t, sto_tensor_nfo_t> _tensor_nfo;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_create_hook;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_delete_hook;

  // the tensors that are preallocated so far
  std::unordered_map<size_t, std::vector<tensor_t*>> _preallocated_tensors;
};

}

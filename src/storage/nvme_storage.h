#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <bits/stdint-intn.h>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <unistd.h>
#include <future>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <list>
#include <queue>
#include <memory>
#include <mutex>
#include "../tensor/tensor.h"
#include "../communication/communicator.h"
#include "../../third_party/cuda/gpu.h"

namespace bbts {

// used to evict tensors
struct lru_t {

  // add a tensor the the lru or updates the current one
  void add(tid_t id) {
    
    // check if we already have it
    auto it = index.find(id);
    if(it != index.end()) {
      queue.erase(it->second);
    }

    // add it to the queue
    queue.push_front(id);
    index[id] = queue.begin();

    // notify that we have more to evict
    cv.notify_all();
  }

  void reassign(tid_t _anon_id, tid_t _id) {

    // set the value
    auto it = index.find(_anon_id);
    *it->second = _id;

    // remove and replace
    index[_id] = it->second;
    index.erase(it);
  }

  // remove it from the lru
  void remove(tid_t id) {

    // remove it from the index and the queue
    auto it = index.find(id);
    if(it != index.end()) {
      queue.erase(it->second);
      index.erase(it);
    }
  }

  // clear the lru
  void clear() {
    index.clear();
    queue.clear();
  }

  // evict one tensor
  tid_t evict(std::unique_lock<std::mutex> &lck) {

    // wait until we have something
    cv.wait(lck, [&]{ return !queue.empty(); });

    // evict a tensor from the queue
    auto id = queue.back(); queue.pop_back();

    // remove it from the index
    index.erase(id);
    
    // return it
    return id;
  }

  // keeps track of the order
  std::list<tid_t> queue;

  // keeps track of the node in the list
  std::unordered_map<tid_t, std::list<tid_t>::iterator> index;

  // used to nofiy that there is more to evict
  std::condition_variable cv;
};

// this class is responsible for memory managment like getting new 
// memory for tensors, getting already existing tensors and in the future evicting
// tensors to disk or moving them to GPU
struct nvme_storage_t {

  nvme_storage_t(communicator_ptr_t com) : _com(std::move(com)) {

    // just empty hooks
    _tensor_create_hook = [](tid_t _) {};
    _tensor_delete_hook = [](tid_t _) {};
    
    // open the file for reading and writing
    _fd = open("./tmp.ts", O_CREAT | O_TRUNC | O_RDWR, 0777);

    // bootstrap the managed memory
    if constexpr(static_config::enable_gpu) {
      #ifdef ENABLE_GPU
      void *ts;
      checkCudaErrors(cudaMallocManaged(&ts, 1024));
      cudaFree(ts);
      #endif
    }
  }

  nvme_storage_t(communicator_ptr_t com, 
                 size_t max_allocated, 
                 const std::string &file) :  _com(std::move(com)), 
                                             _max_allocated(max_allocated) {

    // just empty hooks
    _tensor_create_hook = [](tid_t _) {};
    _tensor_delete_hook = [](tid_t _) {};

    // open the file for reading and writing
    _fd = open(file.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0777);

    // bootstrap the managed memory
    if constexpr(static_config::enable_gpu) {
      #ifdef ENABLE_GPU
      void *ts;
      checkCudaErrors(cudaMallocManaged(&ts, 1024));
      cudaFree(ts);
      #endif
    }
  }

  // destructor
  ~nvme_storage_t();

  // is a reference of the tensor
  struct tensor_ref_t { 

    // the identifier of the tensor
    tid_t id;

    // tensor 
    tensor_t *tensor;
  };

  // the result of a reservation
  struct reservation_result_t {

    reservation_result_t(size_t n, size_t m) { get.reserve(n); create.reserve(m); }

    // the existing tensors we want to get
    std::vector<std::shared_future<tensor_ref_t>> get;

    // the tensors we want to create
    std::vector<std::shared_future<tensor_ref_t>> create;
  };

  // make sure that all the tensors created or requested are aquired at the same time
  template<class fn>
  void local_transaction(const std::vector<tid_t> &get, 
                         std::vector<std::tuple<tid_t, size_t>> create,
                         const fn &fun) {
  
    for(;;) {

      // lock this thing
      std::unique_lock<std::mutex> lck (_m);

      // try to aquire a reservation
      bool val = _try_reserve(get, create);

      if(!val) {

        // maybe I will need to adjust this if it is firing to much
        lck.unlock();
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));

        // try again
        continue;
      }

      // craete the reserved
      auto c = _create_reserved(get, create);

      // run the function
      lck.unlock();
      fun(c.get());
      lck.lock();
      
      // release the reserved
      _release_reservation(get, create);

      // we are done here
      break;
    }
  }

  // if there is a 
  template<class fn>
  void remote_transaction(command_id_t cmd,
                          const bbts::command_t::node_list_t &nodes,
                          const std::vector<tid_t> &get, 
                          std::vector<std::tuple<tid_t, size_t>> create,
                          const fn &fun) {
    
    for(;;) {

      // lock this thing
      std::unique_lock<std::mutex> lck (_m);

      // try to aquire a reservation
      bool val = _try_reserve(get, create);

      // sync all the nodes so that they know if the reservation was aquired or not
      lck.unlock();
      bool res = _com->sync_resource_aquisition(cmd, nodes, val);
      lck.lock();

      // did all the nodes aquire a result if so we can proceed
      if(res) {

        // craete the reserved
        auto c = _create_reserved(get, create);

        // run the function
        lck.unlock();
        fun(c.get());
        lck.lock();

        // release the reserved
        _release_reservation(get, create);

        return;
      }

      // release the reserved if necessary
      if(val) {
        _cancel_reservation(get, create);
      }

      // maybe I will need to adjust this if it is firing to much
      lck.unlock();
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
  }

    // if there is a 
  template<class fn>
  void remote_transaction_p2p(command_id_t cmd,
                              node_id_t other,
                              const std::vector<tid_t> &get, 
                              std::vector<std::tuple<tid_t, size_t>> create,
                              const fn &fun) {
    
    for(;;) {

      // lock this thing
      std::unique_lock<std::mutex> lck (_m);

      // try to aquire a reservation
      bool val = _try_reserve(get, create);
 
      // sync all the nodes so that they know if the reservation was aquired or not
      lck.unlock();
      bool res = _com->sync_resource_aquisition_p2p(cmd, other, val);
      lck.lock();

      // did all the nodes aquire a result if so we can proceed
      if(res) {

        // craete the reserved
        auto c = _create_reserved(get, create);

        // run the function
        lck.unlock();
        fun(c.get());
        lck.lock();

        // release the reserved
        _release_reservation(get, create);

        return;
      }

      // release the reserved if necessary
      if(val) {
        _cancel_reservation(get, create);
      }

      // maybe I will need to adjust this if it is firing to much
      lck.unlock();
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
  }

  // eviction thread
  void request_thread();

  // remove by tid
  bool remove_by_tid(tid_t _id);

  // assign a tid ot the anonymous tensor
  bool assign_tid(tid_t _anon_id, tid_t _id);

  // set the maximum storage
  void set_max_storage(size_t val);

  // does the tensor exist in the storage
  bool has_tensor(tid_t _id);

  // print the nvme_storage
  void print(std::stringstream &ss);

  // clear the nvme_storage
  void clear();

  // shutdown the storage
  void shutdown();

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

  // extract the meta
  std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> extract_meta();

private:

  // we use this to identify the transasction
  using transaction_id = int64_t;

  // the state of the tensor
  enum class tensor_state_t {

    LOADED,
    NOT_LOADED,
    LOADING,
    UNLOADING,
    DELETED,
    REASSIGNED
  };

  // information about the stored tensor
  struct sto_tensor_nfo_t {

    // id of the tensor
    tid_t id;
    
    // the size of the tensor in bytes
    size_t num_bytes = 0;

    // the number of references to this tensor
    size_t num_ref = 0;

    // where did we dump this tensor if we did?
    int64_t file_offset = -1;

    // when we allocate it stays loaded
    tensor_state_t state = tensor_state_t::LOADED;

    // this will be set if the tensor is loaded
    std::shared_future<tensor_ref_t> data;

    // we set the value here
    std::promise<tensor_ref_t> promise;
  };

  // information about the tensors
  struct reservation_nfo_t {

    // the tensors we want to get
    const std::vector<tid_t> *get;

    // the tensors we want to create
    const std::vector<std::tuple<tid_t, size_t>> *create;
  };

  // maps to the information
  std::unordered_map<tid_t, sto_tensor_nfo_t> _tensor_nfo;

  // this reserves space for the tensors in get to be loaded and tensors in create to created
  bool _try_reserve(const std::vector<tid_t> &get,
                    std::vector<std::tuple<tid_t, size_t>> &create);

  // craete all the tensors we just reserved
  std::future<nvme_storage_t::reservation_result_t> _create_reserved(const std::vector<tid_t> &get,
                                                                     const std::vector<std::tuple<tid_t, size_t>> &create);

  // call this to release a already reserved request
  void _release_reservation(const std::vector<tid_t> &get,
                            const std::vector<std::tuple<tid_t, size_t>> &create);

  // call this to cancel a request that was not reserved
  void _cancel_reservation(const std::vector<tid_t> &get,
                           const std::vector<std::tuple<tid_t, size_t>> &create);

  // evict some tensors until we have the required 
  void _evict_some(std::unique_lock<std::mutex> &lck, size_t required);

  // allocate the tensor
  tensor_t *_allocate_tensor(size_t num_bytes);

  // free the allocated tensor
  void free_tensor(tensor_t *tensor);

  // the mutex to lock this thing as it is going to be hammered by threads
  std::mutex _m;

  // currently reserved memory
  size_t _cur_reserved = 0;

  // maximum reserved we are allowed to allocate
  size_t _max_allocated = 0;

  // currently allocated memory if this goes above the maximum allocated we need to move some tensors to disk
  size_t _cur_allocated = 0;

  // evicts pages 
  lru_t _lru;

  // the file we dump stuff into
  int32_t _fd;

  // last offset
  int64_t _file_offset = 0;

  // is the storage shutdown
  bool _is_shutdown = false;

  // all the scheduled reservations
  std::queue<std::tuple<std::promise<reservation_result_t>, reservation_nfo_t>> _scheduled_reservations;

  // 
  std::condition_variable _res_processing_cv;

  // init it to the first available negative number
  tid_t _current_anon = TID_NONE - 1;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_create_hook;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_delete_hook;

  // communicator used for remote transactions
  bbts::communicator_ptr_t _com;
};

}

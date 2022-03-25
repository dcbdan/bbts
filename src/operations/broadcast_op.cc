#include "broadcast_op.h"
#include <cassert>
#include <cstdlib>

namespace bbts {

broadcast_op_t::broadcast_op_t(bbts::communicator_t &_comm,
                               bbts::storage_t &_storage,
                               const bbts::command_t::node_list_t &_nodes,
                               int32_t _tag,
                               size_t _num_bytes,
                               bbts::tid_t _tid): _comm(_comm),
                                                  _storage(_storage),
                                                  _nodes(_nodes),
                                                  _tag(_tag),
                                                  _num_bytes(_num_bytes),
                                                  _tid(_tid) {}

int32_t broadcast_op_t::get_num_nodes() const {
  return _nodes.size();
}

int32_t broadcast_op_t::get_local_rank() const {
  auto it = _nodes.find(_comm.get_rank());
  return it.distance_from(_nodes.begin());
}

int32_t broadcast_op_t::get_global_rank(int32_t local_rank) const {
  return _nodes[local_rank];
}

// runs the broadcast
void broadcast_op_t::apply() {

  int size = get_num_nodes();
  int rank = get_local_rank();
  int vrank = (rank + size ) % size;

  int dim = opal_cube_dim(size);
  int hibit = opal_hibit(vrank, dim);
  --dim;

  // figure out what we need to do
  std::vector<tid_t> get;
  std::vector<std::tuple<tid_t, size_t>> create;
  if(vrank == 0) {
    get = { _tid };
  }
  else {
    create = {{_tid, _num_bytes }};
  }

  // init a remote transaction on all nodes
  bool success = true;
  _storage.remote_transaction(_tag, _nodes, get, create,
  [&](const storage_t::reservation_result_t &res) {

    // the root node has the vrank 0, if this is not the root node
    // we need to recieve the broadcasted tensor
    if (vrank > 0) {

      // figure out the node we need to recieve the data from
      assert(hibit >= 0);
      int peer = ((vrank & ~(1 << hibit))) % size;

      // allocate a buffer for the tensor
      _in = res.create_or_get[0].get().tensor;

      // recieve the request and check if there is an error
      if (!_comm.receive_request_sync(get_global_rank(peer), _tag, _in, _num_bytes)) {
        std::cout << "Failed to recieve the tensor for node " << _comm.get_rank() << " \n";
      }
    }
    else {

      // get the tensors we need to
      _in = res.get[0].get().tensor;
    }

    // allocate the requests
    std::vector<communicator_t::async_request_t> requests;
    requests.reserve(size);

    // send the tensor to the right nodes
    for (int i = hibit + 1, mask = 1 << i; i <= dim; ++i, mask <<= 1) {
      int peer = vrank | mask;
      if (peer < size) {

        // figure out where we need to send it
        peer = peer % size;

        // send the tensor async
        requests.emplace_back(_comm.send_async(_in, _num_bytes, get_global_rank(peer), _tag));

        // we failed here return null
        if(!requests.back().request) {
          std::cout << "Failed to forward a tensor\n";
          exit(-1);
        }
      }
    }

    // wait on all requests
    bool success = true;
    if (!requests.empty()) {
      for(auto &r : requests) {

        // wait for request to finish
        if(!_comm.wait_async(r)) {
          success = false;
        }
      }
    }
  });

  if(!success) {
    std::cout << "Failed to forward a tensor\n";
    exit(-1);
  }
}

}

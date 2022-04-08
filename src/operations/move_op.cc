#include "move_op.h"
#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace bbts {

  // constructs the reduce operation
  move_op_t::move_op_t(bbts::communicator_t &_comm, command_id_t _cmd_id,
                       size_t _num_bytes, tid_t _tid,
                       bool _is_sender, bbts::storage_t &_storage, bbts::node_id_t _node) : _comm(_comm),
                                                                                            _cmd_id(_cmd_id),
                                                                                            _num_bytes(_num_bytes),
                                                                                            _tid(_tid),
                                                                                            _is_sender(_is_sender),
                                                                                            _storage(_storage),
                                                                                            _node(_node) {}

  // apply this operation
  void move_op_t::apply() {

    // is this the sender, if so we initiate a send request
    if(_is_sender) {

      // do the sending
      _storage.remote_transaction_p2p(_cmd_id, _node, {_tid}, {},
      [&](const storage_t::reservation_result_t &res) {

        auto out = res.get[0].get().tensor;
        if(!_comm.send_sync(out, _num_bytes, _node, _cmd_id)) {
          std::cout << "Failed to send the tensor, in a MOVE operation.\n";
          exit(-1);
        }
      });

    } else {

      // do the recieving
      _storage.remote_transaction_p2p(_cmd_id, _node, {}, {{_tid, _num_bytes}},
      [&](const storage_t::reservation_result_t &res) {

        // allocate a buffer for the tensor
        auto out = res.create_or_get.front().get().tensor;

        // recieve the request and check if there is an error
#ifdef ENABLE_IB
        if (!_comm.recv_sync(out, _num_bytes, _node, _cmd_id)) {
          std::cout << "Failed to recieve the tensor, in a MOVE operation.\n";
          exit(-1);
        }
#else
        if (!_comm.receive_request_sync(_node, _cmd_id, out, _num_bytes)) {
          std::cout << "Failed to recieve the tensor, in a MOVE operation.\n";
          exit(-1);
        }
#endif
      });
    }
  }

}

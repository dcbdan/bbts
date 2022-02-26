#pragma once

#include <iostream>

#include "../server/node_config.h"
#include "../ud_functions/ud_function.h"
#include "../tensor/tensor.h"
#include "../commands/command.h"
#include "../server/coordinator_ops.h"

#include "infiniband/connection.h"
#include "infiniband/mr_bytes.h"

struct _silly {
  _silly(std::string s): s(s) {
    std::cout << s << " [enter]" << std::endl;
  }
  ~_silly() {
    std::cout << s << " [exit]" << std::endl;
  }
  std::string s;
};
#define _IBC_COUT_(x) // _silly _silly_monsters_attacking_turtles(x)

namespace bbts {

using ib::tag_t;

struct ib_communicator_t {
  explicit ib_communicator_t(node_config_ptr_t const& _cfg);

  ~ib_communicator_t();

  // send a response string to node 0
  bool send_response_string(const std::string &val);
  std::tuple<bool, std::string> expect_response_string(node_id_t _node);

  // sends, recives a blob with the matching tag from a given node, blocking
  bool send_sync(const void *_bytes, size_t num_bytes, node_id_t _node, int32_t _tag);
  bool recv_sync(void *_bytes, size_t num_bytes, node_id_t _node, int32_t _tag);
  // and non blocking
  std::future<bool> send_async(
    const void *_bytes, size_t num_bytes, node_id_t _node, int32_t _tag);

  // notify a node that tensors were created
  bool tensors_created_notification(
    node_id_t out_node, const std::vector<bbts::tid_t> &tensor);
  // wait to receive a notification
  std::tuple<node_id_t, std::vector<bbts::tid_t>> receive_tensor_created_notification();
  // shutdown the notification handler
  bool shutdown_notification_handler();

  // initiates the operation on all the specified nodes
  bool op_request(const command_ptr_t &_cmd);
  // waits to recieve an operation
  command_ptr_t expect_op_request();
  // this sends a shutdown command to the thread that is calling @see expect_op_request
  bool shutdown_op_request();

  // send the coord op to all nodes
  bool send_coord_op(const bbts::coordinator_op_t &op);
  // expect the a coord op
  bbts::coordinator_op_t expect_coord_op();

  // send the cmds to all nodes
  bool send_coord_cmds(const std::vector<command_ptr_t> &cmds);

  // expect the a coord op
  bool expect_coord_cmds(size_t num_cmds, std::vector<command_ptr_t> &out);

  bool send_tensor_meta(const std::vector<std::tuple<tid_t, tensor_meta_t>> &meta);
  bool recv_meta(node_id_t node, std::vector<std::tuple<tid_t, tensor_meta_t>> &data);

  // sync the resource aquisition
  bool sync_resource_aquisition(
    command_id_t cmd,
    const bbts::command_t::node_list_t &nodes,
    bool my_val);

  // sync the resource aquisition between two nodes
  bool sync_resource_aquisition_p2p(command_id_t cmd, node_id_t &node, bool my_val);

  // send and recv the tensors size
  bool send_tensor_size(node_id_t node, int32_t tag, uint64_t val);
  std::tuple<uint64_t, bool> recv_tensor_size(node_id_t node, int32_t tag);

  // send a bunch of bytes to all nodes
  bool send_bytes(char* file, size_t file_size);
  bool expect_bytes(size_t num_bytes, std::vector<char> &out);

  // Create a memory region object. If this method is called, then it is assumed
  // that future calls to
  //   send_sync,
  //   send_async and
  //   recv_sync
  // will contain a memory segment from the returned memory region.
  // Otherwise everything will break and tensors won't be sent and recvd
  // correctly.
  //
  // If there is only one node, it just allocates the memory,
  // not create a memory region
  std::shared_ptr<ib::memory_region_bytes_t>
    create_and_use_tensor_memory_region(size_t num_bytes);

  // waits for all the nodes to hit this, should only be used for initialization
  void barrier();

  // return the rank
  int32_t get_rank() const;

  // return the number of nodes
  int32_t get_num_nodes() const;

private:
  enum com_tag {
    response_string_tag,
    notify_tensor_tag,
    send_cmd_tag,
    coordinator_tag,
    coordinator_bcast_cmd_tag,
    coordinator_bcast_bytes,
    barrier_tag,
    tensor_meta_tag,
    free_tag
  };

private:
  ib::connection_t connection;
  std::shared_ptr<ib::memory_region_bytes_t> tensor_memory;
};

} // bbts


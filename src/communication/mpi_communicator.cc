#include "communicator.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <mpi.h>
#include <unistd.h>

namespace bbts {

mpi_communicator_t::mpi_communicator_t(const node_config_ptr_t &_cfg) {

  // initialize the mpi
  int32_t provided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE not provided");
  }

  // get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &_num_nodes);

  // get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
}

mpi_communicator_t::~mpi_communicator_t() {

  // shutdown the communicator
  MPI_Finalize();
}

// send a response string
bool mpi_communicator_t::send_response_string(const std::string &val) {

  // get the number of byte to send and send the request
  return MPI_Ssend(val.c_str(), val.size(), MPI_CHAR, 0, RESPONSE_STRING_TAG, MPI_COMM_WORLD) == MPI_SUCCESS;
}

// expect a response string
std::tuple<bool, std::string> mpi_communicator_t::expect_response_string(node_id_t _node) {

  // wait for a request
  sync_request_t _req;
  auto mpi_errno = MPI_Mprobe(_node, RESPONSE_STRING_TAG, MPI_COMM_WORLD, &_req.message, &_req.status);

  // check for errors
  if (mpi_errno != MPI_SUCCESS) {        
      return {false, ""};
  }

  // get the size
  MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);

  // allocate the memory and receive the string
  std::unique_ptr<char[]> p(new char[_req.num_bytes]);
  if(MPI_Mrecv (p.get(), _req.num_bytes, MPI_CHAR, &_req.message, &_req.status) != MPI_SUCCESS) {
    return {false, ""};
  }

  // return it
  return {true, std::string(p.get(), _req.num_bytes)};
}

bool mpi_communicator_t::recv_sync(void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag) {

  // recive the stuff
  return MPI_Recv(_bytes, num_bytes, MPI_CHAR, _node, _tag + FREE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
}

// does the send, method is blocking
bool mpi_communicator_t::send_sync(const void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag) {

  // get the number of byte to send and send the request
  return MPI_Ssend(_bytes, num_bytes, MPI_CHAR, _node, _tag + FREE_TAG, MPI_COMM_WORLD) == MPI_SUCCESS;
}

bool mpi_communicator_t::wait_async(mpi_communicator_t::async_request_t &_request) {
  return MPI_Wait(&_request.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS;
}

bool mpi_communicator_t::tensors_created_notification(node_id_t out_node, const std::vector<bbts::tid_t> &tensor) {
  return MPI_Ssend(tensor.data(), (int32_t) (tensor.size() * sizeof(bbts::tid_t)), MPI_CHAR, out_node, NOTIFY_TENSOR_TAG, MPI_COMM_WORLD) == MPI_SUCCESS;
}

std::tuple<node_id_t, std::vector<bbts::tid_t>> mpi_communicator_t::receive_tensor_created_notification() {

  // wait for a request
  sync_request_t _req; int32_t flag = false;
  int32_t mpi_errno;
  while (!flag) {
    mpi_errno = MPI_Improbe(ANY_NODE, NOTIFY_TENSOR_TAG, MPI_COMM_WORLD, &flag, &_req.message, &_req.status);
    usleep(100);
  }

  // check for errors
  if(mpi_errno != MPI_SUCCESS) {
    return {-1, {} };
  }

  // get the size  and set the tag for the request
  MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);

  // allocate the memory and receive the command
  std::vector<bbts::tid_t> p(_req.num_bytes / sizeof(bbts::tid_t));
  if(MPI_Mrecv (p.data(), _req.num_bytes, MPI_CHAR, &_req.message, &_req.status) != MPI_SUCCESS) {
    return {-1, {}};;
  }

  return { _req.status.MPI_SOURCE, std::move(p) };
}

bool mpi_communicator_t::shutdown_notification_handler() {

  // just a tensor with a tid -1
  std::vector<bbts::tid_t> tensor = { -1 };
  return MPI_Ssend(tensor.data(), (int32_t) (tensor.size() * sizeof(bbts::tid_t)), MPI_CHAR,
                   get_rank(), NOTIFY_TENSOR_TAG, MPI_COMM_WORLD) == MPI_SUCCESS;
}

mpi_communicator_t::async_request_t mpi_communicator_t::send_async(const void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag) {

  // initiate an asynchronous send request
  async_request_t _req;
  _req.success = MPI_Isend(_bytes, num_bytes, MPI_CHAR, _node, _tag + FREE_TAG, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;

  // return the request handle
  return _req;
}

mpi_communicator_t::sync_request_t mpi_communicator_t::expect_request_sync(node_id_t _node, com_tags _tag) {

  // wait for a request
  sync_request_t _req;
  auto mpi_errno = MPI_Mprobe(_node, _tag + FREE_TAG, MPI_COMM_WORLD, &_req.message, &_req.status);

  // check for errors
  if (mpi_errno != MPI_SUCCESS) {        
      _req.success = false;
      return _req;
  }

  // get the size
  MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);

  // set the message type
  _req.message_tag = (com_tags) _req.status.MPI_TAG;

  // return the request
  return _req;
}

// recieves the request that we got from expect_request_sync
bool mpi_communicator_t::receive_request_sync(node_id_t node, com_tags tag, void *bytes, size_t num_bytes) {

  // recieve the requests
  MPI_Status status;
  return MPI_Recv(bytes, num_bytes, MPI_CHAR, node, tag + FREE_TAG, MPI_COMM_WORLD, &status) == MPI_SUCCESS;
}

bool mpi_communicator_t::op_request(const command_ptr_t &_cmd) {

  // find all the nodes referenced in the input
  std::vector<node_id_t> to_sent_to;
  auto nodes = _cmd->get_nodes();
  for(int node : nodes) {
    if(node != _rank) {
      to_sent_to.push_back(node);
    }
  }

  // initiate an asynchronous send request
  std::vector<async_request_t> requests;
  for(auto node : to_sent_to) {
    async_request_t _req;
    _req.success = MPI_Isend(_cmd.get(), _cmd->num_bytes(), MPI_CHAR, node, SEND_CMD_TAG, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;
    requests.push_back(_req);
  }

  // wait for all the requests to finish
  bool success = true;
  for(auto &r : requests) {
    success = r.success && MPI_Wait(&r.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS && success;
  }

  return success;
}

bool mpi_communicator_t::shutdown_op_request() {

  // create a shutdown command to send to the remote handler
  command_ptr_t cmd = command_t::create_shutdown(get_rank());

  // initiate an asynchronous send request
  std::vector<async_request_t> requests;

  // init the send
  async_request_t _req;
  _req.success = MPI_Isend(cmd.get(), cmd->num_bytes(), MPI_CHAR,
                           get_rank(), SEND_CMD_TAG, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;

  // wait for all the request to finish
  return _req.success && MPI_Wait(&_req.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS;;
}

command_ptr_t mpi_communicator_t::expect_op_request() {

  // wait for a request
  sync_request_t _req; int32_t flag = false;
  int32_t mpi_errno;
  while (!flag) {
    mpi_errno = MPI_Improbe(ANY_NODE, SEND_CMD_TAG, MPI_COMM_WORLD, &flag, &_req.message, &_req.status);
    usleep(100);
  }

  // check for errors
  if(mpi_errno != MPI_SUCCESS) {
    return nullptr;
  }

  // get the size  and set the tag for the request
  MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);
  _req.message_tag = (com_tags) _req.status.MPI_TAG;

  // allocate the memory and receive the command
  std::unique_ptr<char[]> p(new char[_req.num_bytes]);
  if(MPI_Mrecv (p.get(), _req.num_bytes, MPI_CHAR, &_req.message, &_req.status) != MPI_SUCCESS) {
    return nullptr;
  }

  // cast it to the command
  auto p_rel = p.release();
  auto p_cmd = (bbts::command_t *)(p_rel);
  auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(p_cmd);

  // move the command
  return std::move(d);
}

bool mpi_communicator_t::sync_resource_aquisition(command_id_t cmd, const bbts::command_t::node_list_t &nodes, bool my_val) {

  // get the world group
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // the group that is syncing on the resource
  MPI_Group resource_group;
  MPI_Group_incl(world_group, nodes.size(), nodes.data(), &resource_group);

  // the resource group
  MPI_Comm resource_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, resource_group, cmd + FREE_TAG, &resource_comm);
  
  // get the result
  bool out;
  MPI_Allreduce(&my_val, &out, 1, MPI_C_BOOL, MPI_LAND, resource_comm);

  // do a reduce on the value
  MPI_Group_free(&world_group);
  MPI_Group_free(&resource_group);
  MPI_Comm_free(&resource_comm);

  // return the result
  return out;
}

bool mpi_communicator_t::sync_resource_aquisition_p2p(command_id_t cmd, node_id_t &node, bool my_val) {

  // the requests
  MPI_Request send_request_lr, recv_request_lr;

  // exchange the values
  bool its_val;
  MPI_Isend(&my_val,  1, MPI_C_BOOL, node,  cmd + FREE_TAG, MPI_COMM_WORLD, &recv_request_lr);
  MPI_Irecv(&its_val, 1, MPI_C_BOOL, node, cmd + FREE_TAG, MPI_COMM_WORLD, &send_request_lr);

  // sync everything
  MPI_Status status;
  MPI_Wait(&send_request_lr, &status);
  MPI_Wait(&recv_request_lr, &status);

  // return the result
  return its_val && my_val;
}

// waits for all the nodes to hit this, should only be used for initialization
void mpi_communicator_t::barrier() {

  // wait for every node
  MPI_Barrier(MPI_COMM_WORLD);
}

bool mpi_communicator_t::send_coord_op(const bbts::coordinator_op_t &op) {

  // initiate an asynchronous send request
  std::vector<async_request_t> requests; requests.reserve(_num_nodes);
  for(node_id_t node = 1; node < _num_nodes; ++node) {
    async_request_t _req;
    _req.success = MPI_Isend(&op, sizeof(op), MPI_CHAR, node, COORDINATOR_TAG, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;
    requests.push_back(_req);
  }

  // wait for all the requests to finish
  bool success = true;
  for(auto &r : requests) {
    success = r.success && MPI_Wait(&r.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS && success;
  }

  return success;
}

bool mpi_communicator_t::send_coord_op(const bbts::coordinator_op_t &op, node_id_t node) {

  // initiate an asynchronous send request
  async_request_t _req;
  _req.success = MPI_Isend(&op, sizeof(op), MPI_CHAR, node, COORDINATOR_TAG, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;

  // wait for all the requests to finish
  _req.success = MPI_Wait(&_req.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS && _req.success;
  return _req.success;
}

bbts::coordinator_op_t mpi_communicator_t::expect_coord_op() {

  // wait for a request
  sync_request_t _req;
  auto mpi_errno = MPI_Mprobe(ANY_NODE, COORDINATOR_TAG, MPI_COMM_WORLD, &_req.message, &_req.status);

  // check for errors
  bbts::coordinator_op_t op{};
  if(mpi_errno != MPI_SUCCESS) {
    op._type = coordinator_op_types_t::FAIL;
    return op;
  }

  // get the size  and set the tag for the request
  MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);
  _req.message_tag = (com_tags) _req.status.MPI_TAG;

  // allocate the memory and receive the command
  if(MPI_Mrecv (&op, sizeof(op), MPI_CHAR, &_req.message, &_req.status) != MPI_SUCCESS) {
    op._type = coordinator_op_types_t::FAIL;
    return op;
  }

  // move the command
  return op;
}

// send the cmds to all nodes
bool mpi_communicator_t::send_coord_cmds(const std::vector<command_ptr_t> &cmds) {

  // send all the commands
  for(auto &cmd : cmds) {

    // initiate an asynchronous send request
    std::vector<async_request_t> requests; requests.reserve(_num_nodes);
    for(node_id_t node = 1; node < _num_nodes; ++node) {
      async_request_t _req;
      _req.success = MPI_Isend(cmd.get(), cmd->num_bytes(), MPI_CHAR, node, COORDINATOR_BCAST_CMD_TAG, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;
      requests.push_back(_req);
    }

    // wait for all the requests to finish
    bool success = true;
    for(auto &r : requests) {
      success = r.success && MPI_Wait(&r.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS && success;
    }

    // make sure we succeeded
    if(!success) {
      return false;
    }
  }

  return true;
}

bool mpi_communicator_t::send_bytes(char* file, size_t file_size) {
  // send it everywhere except the root node
  std::vector<async_request_t> requests; requests.reserve(_num_nodes);
  for(node_id_t node = 1; node < _num_nodes; ++node) {
    async_request_t _req;
    _req.success = MPI_Isend(file, file_size, MPI_CHAR, node, COORDINATOR_BCAST_BYTES, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;
    requests.push_back(_req);
  }

  bool success = true;
  for(auto &r : requests) {
    success = r.success && MPI_Wait(&r.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS && success;
  }

  return success;  
}

// send a bunch of bytes to all nodes
bool mpi_communicator_t::send_tensor_meta(const std::vector<std::tuple<tid_t, tensor_meta_t>> &meta) {

  // the number of bytes
  size_t num_bytes = meta.size() * sizeof(std::tuple<tid_t, tensor_meta_t>);

  // send the size needed to recieve
  if(MPI_Ssend(&num_bytes, sizeof(num_bytes), MPI_CHAR, 0, TENSOR_META_TAG_SIZE, MPI_COMM_WORLD) != MPI_SUCCESS) {
    return false;
  }

  // get the number of byte to send and send the request
  return MPI_Ssend(meta.data(), meta.size() * sizeof(std::tuple<tid_t, tensor_meta_t>), MPI_CHAR, 0, TENSOR_META_TAG, MPI_COMM_WORLD) == MPI_SUCCESS;
}

// get the meta from a node
bool mpi_communicator_t::recv_meta(node_id_t node, std::vector<std::tuple<tid_t, tensor_meta_t>> &data) {

  // recieve the tensor meta
  size_t num_bytes;
  if(MPI_Recv(&num_bytes, sizeof(num_bytes), MPI_CHAR, node, TENSOR_META_TAG_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
    return false;
  }

  // resize the data
  data.resize(num_bytes / sizeof(std::tuple<tid_t, tensor_meta_t>));
  return MPI_Recv(data.data(), num_bytes, MPI_CHAR, node, TENSOR_META_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
}

// expect the a coord op
bool mpi_communicator_t::expect_coord_cmds(size_t num_cmds, std::vector<command_ptr_t> &out) {

  out.reserve(num_cmds);
  for(size_t i = 0; i < num_cmds; ++i) {

    // wait for a request
    sync_request_t _req;
    auto mpi_errno = MPI_Mprobe(ANY_NODE, COORDINATOR_BCAST_CMD_TAG, MPI_COMM_WORLD, &_req.message, &_req.status);

    // check for errors
    if(mpi_errno != MPI_SUCCESS) {
      return false;
    }

    // get the size  and set the tag for the request
    MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);
    _req.message_tag = (com_tags) _req.status.MPI_TAG;

    // allocate the memory and receive the command
    std::unique_ptr<char[]> p(new char[_req.num_bytes]);
    if(MPI_Mrecv (p.get(), _req.num_bytes, MPI_CHAR, &_req.message, &_req.status) != MPI_SUCCESS) {
      return false;
    }

    // cast it to the command
    auto p_rel = p.release();
    auto p_cmd = (bbts::command_t *)(p_rel);
    auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(p_cmd);

    // store the command
    out.push_back(std::move(d));
  }

  return true;
}

bool mpi_communicator_t::expect_bytes(size_t num_bytes, std::vector<char> &out) {
  
  // recive the stuff
  out.resize(num_bytes);
  return MPI_Recv(out.data(), num_bytes, MPI_CHAR, ANY_NODE, COORDINATOR_BCAST_BYTES, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
}

// return the rank
int32_t mpi_communicator_t::get_rank() const {
  return _rank;
}

// return the number of nodes
int32_t mpi_communicator_t::get_num_nodes() const {
  return _num_nodes;
}


}

#include "ib_communicator.h"

#include <iostream>
#include <fstream>

namespace bbts {

using namespace ib;

bool wait_all_bools(std::vector<std::future<bool>>& futs) {
  bool success = true;
  for(auto& fut: futs) {
    success = success & fut.get();
  }
  return success;
}

std::tuple<std::string, int32_t, std::vector<std::string>>
parse_connection_args(node_config_ptr_t const& cfg) {
  auto argc = cfg->argc;
  auto argv = cfg->argv;

  std::string usage = "usage: " + std::string(argv[0]) + " <rank> <device name> <hosts file>";
  if(argc != 4) {
    throw std::runtime_error(usage);
  }

  int rank = std::stoi(argv[1]);

  std::vector<std::string> ips;
  {
    std::ifstream s = std::ifstream(argv[3]);
    if(!s.is_open()) {
      usage = "Coud not open '" + std::string(argv[3]) + "'\n" + usage;
      throw std::runtime_error(usage);
    }

    std::string l;
    while(std::getline(s, l)) {
      if(l.size() < 7) {
        usage = "Invalid hosts file\n" + usage;
        throw std::runtime_error(usage);
      }
      ips.push_back(l);
    }
    for(auto i: ips){
      std::cout << "IP: " << i << std::endl;
    }
  }

  return {argv[2], rank, ips};
}

ib_communicator_t::ib_communicator_t(node_config_ptr_t const& _cfg):
    connection(parse_connection_args(_cfg), com_tag::free_tag),
                           // ^ pin all tags before the free tag
    tensor_memory(nullptr)
{}

ib_communicator_t::~ib_communicator_t() {}

// send a response string
bool ib_communicator_t::send_response_string(const std::string &val) {
  _IBC_COUT_("send_response_string");
  if(get_rank() == 0) {
    throw std::runtime_error("node 0 should not send response string");
  }

  auto fut = connection.send(
    0,
    com_tag::response_string_tag,
    to_bytes_t(val.data(), val.size()));

  return fut.get();
}

// expect a response string
std::tuple<bool, std::string>
ib_communicator_t::expect_response_string(
  int32_t from_rank)
{
  _IBC_COUT_("expect_response_string");
  if(get_rank() != 0) {
    throw std::runtime_error("only node 0 should recv response string");
  }
  if(from_rank == 0) {
    throw std::runtime_error("cannot expect message from self");
  }

  auto [success, bytes] = connection.recv_from(from_rank, com_tag::response_string_tag).get();

  std::string ret;
  if(success) {
    ret = std::string(bytes.ptr.get(), bytes.size);
  }
  return {success, ret};
}

bool ib_communicator_t::send_sync(
  const void *bytes,
  size_t num_bytes,
  node_id_t dest_rank,
  int32_t tag)
{
  _IBC_COUT_("send_sync");
  return send_async(bytes, num_bytes, dest_rank, tag).get();
}

bool ib_communicator_t::recv_sync(
  void *bytes, size_t num_bytes,
  node_id_t from_rank,
  int32_t tag)
{
  _IBC_COUT_("recv_sync");
  if(tensor_memory) {
    return connection.recv_from_with_bytes(
      from_rank,
      com_tag::free_tag + tag,
      {bytes, num_bytes},
      tensor_memory->get_memory_region()
    ).get();
  } else {
    return connection.recv_from_with_bytes(
      from_rank,
      com_tag::free_tag + tag,
      {bytes, num_bytes}
    ).get();
  }
}

std::future<bool> ib_communicator_t::send_async(
  const void *bytes,
  size_t num_bytes,
  node_id_t dest_rank,
  int32_t tag)
{
  _IBC_COUT_("send_async");
  if(tensor_memory) {
    return connection.send(
      dest_rank,
      com_tag::free_tag + tag,
      {(void*)bytes, num_bytes},
      tensor_memory->get_memory_region());
  } else {
    return connection.send(
      dest_rank,
      com_tag::free_tag + tag,
      {(void*)bytes, num_bytes});
  }
}

bool ib_communicator_t::tensors_created_notification(
  node_id_t dest_rank,
  const std::vector<bbts::tid_t> &tensor)
{
  _IBC_COUT_("tensors_created_notification");
  return connection.send(
    dest_rank,
    com_tag::notify_tensor_tag,
    to_bytes_t(tensor.data(), tensor.size())
  ).get();
}

std::tuple<node_id_t, std::vector<bbts::tid_t>>
ib_communicator_t::receive_tensor_created_notification() {
  _IBC_COUT_("recv tensor created notification");

  auto [success, from_rank, bytes] = connection.recv(
    com_tag::notify_tensor_tag
  ).get();

  // just keep doing this method on failure
  if(!success) {
    return receive_tensor_created_notification();
  }

  bbts::tid_t* beg = (bbts::tid_t*)bytes.ptr.get();
  bbts::tid_t* end = (bbts::tid_t*)(bytes.ptr.get() + bytes.size);

  return {from_rank,  std::vector<bbts::tid_t>(beg, end)};
}

bool ib_communicator_t::shutdown_notification_handler() {
  _IBC_COUT_("shutdown_notification_handler");
  std::vector<bbts::tid_t> tensor = { -1 };
  return tensors_created_notification(get_rank(), tensor);
}

bool ib_communicator_t::op_request(const command_ptr_t &_cmd) {
  _IBC_COUT_("op_request");
  // find all the nodes referenced in the input
  std::vector<node_id_t> to_send_to;
  auto nodes = _cmd->get_nodes();
  for(int node : nodes) {
    if(node != get_rank()) {
      to_send_to.push_back(node);
    }
  }

  std::vector<std::future<bool>> futs;
  for(auto dest_rank : to_send_to) {
    futs.push_back(
      connection.send(
        dest_rank,
        com_tag::send_cmd_tag,
        {_cmd.get(), _cmd->num_bytes()}));
  }

  // wait for all the sends to finish
  return wait_all_bools(futs);
}

bool ib_communicator_t::shutdown_op_request() {
  _IBC_COUT_("shutdown_op_request");

  // create a shutdown command to send to the remote handler
  command_ptr_t cmd = command_t::create_shutdown(get_rank());

  return connection.send(
            get_rank(),
            com_tag::send_cmd_tag,
            {cmd.get(), cmd->num_bytes()}).get();
}

command_ptr_t ib_communicator_t::expect_op_request() {
  _IBC_COUT_("expect_op_request");

  // recv a message from anywhere
  auto [success, from_rank, own_bytes] = connection.recv(com_tag::send_cmd_tag).get();

  // check for errors
  if(!success) {
    return nullptr;
  }

  // cast it to the command
  auto p_rel = own_bytes.ptr.release();
  auto p_cmd = (bbts::command_t *)(p_rel);
  auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(p_cmd);

  // move the command
  return std::move(d);
}

bool ib_communicator_t::sync_resource_aquisition(
  command_id_t cmd,
  const bbts::command_t::node_list_t &nodes,
  bool my_val)
{
  throw std::runtime_error("resource acquisition methods not implemented");
  return false;
}

bool ib_communicator_t::sync_resource_aquisition_p2p(
  command_id_t cmd,
  node_id_t &node,
  bool my_val)
{
  throw std::runtime_error("resource acquisition methods not implemented");
  return false;
}

// waits for all the nodes to hit this, should only be used for initialization
void ib_communicator_t::barrier() {
  _IBC_COUT_("barrier");

  // TODO: Implement an bbts::ib::connection_t::barrier
  // TODO: Guessing this will do as a "barrier"
  char c;
  bytes_t bs = to_bytes_t(&c, 1);

  // send bs to every other node, recv from every other node
  std::vector<std::future<bool                    >> send_futs;
  std::vector<std::future<tuple<bool, own_bytes_t>>> recv_futs;
  for(node_id_t node = 0; node < get_num_nodes(); ++node) {
    if(node == get_rank()) {
      continue;
    }
    send_futs.push_back(connection.send(
      node,
      com_tag::barrier_tag,
      bs));
    recv_futs.push_back(connection.recv_from(
      node,
      com_tag::barrier_tag));
  }

  for(int i = 0; i != send_futs.size(); ++i) {
    bool success_send = send_futs[i].get();
    auto [success_recv, _] = recv_futs[i].get();
    if(!success_send || !success_recv) {
      throw std::runtime_error("ib_communicator barrier failed");
    }
  }
}

bool ib_communicator_t::send_coord_op(const bbts::coordinator_op_t &op) {
  _IBC_COUT_("send coord op");
  // send the op to all nodes except node zero
  std::vector<std::future<bool>> futs;
  futs.reserve(get_num_nodes());
  for(node_id_t node = 1; node < get_num_nodes(); ++node) {
    futs.push_back(
      connection.send(
        node,
        com_tag::coordinator_tag,
        to_bytes_t(&op, 1)));
  }

  // wait for all the requests to finish
  return wait_all_bools(futs);
}

bbts::coordinator_op_t ib_communicator_t::expect_coord_op() {
  _IBC_COUT_("expect coord op");
  // recv coord cmds from node zero
  bbts::coordinator_op_t op{};
  bool success = connection.recv_from_with_bytes(
    0,
    com_tag::coordinator_tag,
    to_bytes_t(&op, 1)).get();

  // check for errors
  if(!success) {
    op._type = coordinator_op_types_t::FAIL;
    return op;
  }

  return op;
}

// send the cmds to all nodes
bool ib_communicator_t::send_coord_cmds(const std::vector<command_ptr_t> &cmds) {
  _IBC_COUT_("send coord cmds");
  // send all the commands to all nodes except node 0
  for(auto &cmd : cmds) {
    std::vector<std::future<bool>> futs;
    futs.reserve(get_num_nodes());

    for(node_id_t node = 1; node < get_num_nodes(); ++node) {
      futs.push_back(
        connection.send(
          node,
          com_tag::coordinator_bcast_cmd_tag,
          {cmd.get(), cmd->num_bytes()}));
    }

    if(!wait_all_bools(futs)) {
      return false;
    }
  }

  return true;
}

// expect the a coord op
bool ib_communicator_t::expect_coord_cmds(
  size_t num_cmds,
  std::vector<command_ptr_t> &out)
{
  _IBC_COUT_("expect_coord_cmds");
  // recv one command at a time from node zero and if any fails, stop
  out.reserve(num_cmds);
  for(size_t i = 0; i < num_cmds; ++i) {
    auto [success, own_bytes] = connection.recv_from(
        0,
        com_tag::coordinator_bcast_cmd_tag).get();
    if(success){
      // cast it to the command
      auto p_rel = own_bytes.ptr.release();
      auto p_cmd = (bbts::command_t *)(p_rel);
      auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(p_cmd);

      out.push_back(std::move(d));
    } else {
      return false;
    }
  }
  return true;
}

bool ib_communicator_t::send_tensor_meta(
  std::vector<std::tuple<tid_t, tensor_meta_t>> const& meta)
{
  _IBC_COUT_("send_tensor_meta");

  // send the meta info to node 0

  // the number of bytes
  size_t num_bytes = meta.size() * sizeof(std::tuple<tid_t, tensor_meta_t>);

  return connection.send(
    0,
    com_tag::tensor_meta_tag,
    {(void*)meta.data(), num_bytes}).get();
}

bool ib_communicator_t::recv_meta(
  node_id_t node,
  std::vector<std::tuple<tid_t, tensor_meta_t>> &data)
{
  _IBC_COUT_("recv_meta");
  // TODO: this method does an extra copy
  // One could do two connection.send messages, one with the size. But that'd be silly
  // because the connection class is already sending size information in it's communication.
  // The problem is that std::vector is owning the data and the connection class isn't
  // std::vector aware. The best bet would be to use a data structure besides std::vector..

  auto [success, own_bytes] = connection.recv_from(node, com_tag::tensor_meta_tag).get();

  if(success) {
    auto num = own_bytes.size / sizeof(std::tuple<tid_t, tensor_meta_t>);
    data.resize(num);
    std::tuple<tid_t, tensor_meta_t>* recv_beg =
      (std::tuple<tid_t, tensor_meta_t>*) own_bytes.ptr.get();
    std::copy(recv_beg, recv_beg + num, data.begin());
  }
  return success;
}


// TODO: it should be possible to create a fixed size message connection object.
//       it is a bit silly to send such small messages with connection.h as it is now
bool ib_communicator_t::send_tensor_size(node_id_t node, int32_t tag, uint64_t val) {
  _IBC_COUT_("send tensor size");
  return connection.send(node, com_tag::free_tag + tag, to_bytes_t(&val, 1)).get();
}

std::tuple<uint64_t, bool> ib_communicator_t::recv_tensor_size(node_id_t node, int32_t tag) {
  _IBC_COUT_("recv tensor size");
  auto [success, own_bytes] = connection.recv_from(node, com_tag::free_tag + tag).get();
  uint64_t ret;
  if(success) {
    ret = *((uint64_t*)own_bytes.ptr.release());
  }
  return {ret, success};
}

bool ib_communicator_t::send_bytes(char* file, size_t file_size) {
  _IBC_COUT_("send bytes");
  // send it everywhere except the root node
  std::vector<std::future<bool>> futs;
  futs.reserve(get_num_nodes());

  for(node_id_t node = 1; node < get_num_nodes(); ++node) {
    futs.push_back(
      connection.send(
        node,
        coordinator_bcast_bytes,
        {(void*)file, file_size}));
  }
  return wait_all_bools(futs);
}

bool ib_communicator_t::expect_bytes(size_t num_bytes, std::vector<char> &out) {
  _IBC_COUT_("expect bytes");
  out.reserve(num_bytes);

  bool success = connection.recv_from_with_bytes(
    0,
    com_tag::coordinator_bcast_bytes,
    {(void*)out.data(), num_bytes}).get();

  return success;
}

std::shared_ptr<ib::memory_region_bytes_t>
  ib_communicator_t::create_and_use_tensor_memory_region(size_t num_bytes)
{
  auto ret = std::shared_ptr<ib::memory_region_bytes_t>(new ib::memory_region_bytes_t());

  if(get_num_nodes() > 1) {
    if(!ret->setup_bytes_and_mr(
          num_bytes,
          &connection,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
    {
      throw std::runtime_error("Couldn't allocate and register memory!");
    }
  } else {
    if(!ret->setup_bytes(num_bytes))
    {
      throw std::runtime_error("Couldn't allocate memory!");
    }
  }

  return ret;
}

// return the rank
int32_t ib_communicator_t::get_rank() const {
  return connection.get_rank();
}

// return the number of nodes
int32_t ib_communicator_t::get_num_nodes() const {
  return connection.get_num_nodes();
}

}

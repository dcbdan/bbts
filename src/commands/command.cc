#include "command.h"

namespace bbts {

using tid_node_id_t = command_t::tid_node_id_t;

tid_node_id_t command_t::get_reduce_input(node_id_t _node_id) const
{
  // try to find an input for this node
  auto inputs = get_inputs();
  for(auto in : inputs) {
    if(in.node == _node_id) {
      return in;
    }
  }

  return {-1, -1};
}

bool command_t::is_local_reduce(node_id_t _node_id) const {
  // make sure it is actually a reduce
  if(type != op_type_t::REDUCE) {
    return false;
  }

  // check if the output and all inputs are on the same node
  auto nodes = get_nodes();
  for(int32_t idx = 0; idx < nodes.size(); idx++) {
    if(nodes[idx] != _node_id) { return false; }
  }
  return true;
}

bool command_t::is_remote_reduce(node_id_t _node_id) const {

  // make sure it is actually a reduce
  if(type != op_type_t::REDUCE) {
    return false;
  }

  // if it is not local it is remote
  return !is_local_reduce(_node_id);
}

bool command_t::uses_node(node_id_t target_node) const
{
  // go and check every tensor if it is located on a node
  auto nodes = get_nodes();
  for(auto &node : nodes) {
    if(node == target_node) {
      return true;
    }
  }

  return false;
}

void command_t::print(std::stringstream &ss) {
  ss << "{.id : " << id << " ";
  ss << ".type : ";

  switch (type) {
    case MOVE           : ss << (_num_outputs == 1 ? "MOVE " : "BROADCAST ") ; break;
    case APPLY          : ss << "APPLY ";          break;
    case DELETE         : ss << "DELETE ";         break;
    case REDUCE         : ss << "REDUCE ";         break;
    case TOUCH          : ss << "TOUCH ";          break;
    case SHUTDOWN       : ss << "SHUTDOWN ";       break;
  }

  ss << " .inputs : [";
  for(int32_t idx = 0; idx < _num_inputs; idx++) {
    ss << "( " << get_input(idx).tid << ", " << get_input(idx).node << "),";
  }
  ss << "]";

  ss << " .outputs : [";
  for(int32_t idx = 0; idx < _num_outputs; idx++) {
    ss << "( " << get_output(idx).tid << ", " << get_output(idx).node << "),";
  }
  ss << "]\n";
}

// clone the command
command_ptr_t command_t::clone() {

  // allocate the memory
  std::unique_ptr<char[]> tmp(new char[num_bytes()]);

  // copy everything
  memcpy(tmp.get(), this, num_bytes());

  // return the the clone
  auto pReleased = tmp.release();
  auto pDerived = (bbts::command_t *)(pReleased);
  auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(pDerived);

  // move the command
  return std::move(d);
}

// allocate the command
// static
command_ptr_t command_t::allocate_command(size_t num_bytes) {

  // allocate the memory
  std::unique_ptr<char[]> p(new char[num_bytes]);

  auto pReleased = p.release();
  auto pDerived = (bbts::command_t *)(pReleased);
  auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(pDerived);

  // move the command
  return std::move(d);
}

// static
command_ptr_t command_t::create_move(
  command_id_t id,
  tid_node_id_t in,
  tid_node_id_t out)
{
  // make sure this matches
  assert(in.tid == out.tid);

  // create the output
  auto tmp = allocate_command(_num_bytes(0, 2, 1, 1));

  // set the id type and function
  tmp->id = id;
  tmp->type = MOVE;
  tmp->fun_id = {-1, -1};
  tmp->_num_parameters = 0;
  tmp->_num_nodes = 2;
  tmp->_num_inputs = 1;
  tmp->_num_outputs = 1;

  // setup the offsets
  tmp->_setup_offsets();

  // fill-up the nodes
  tmp->_nodes()[0] = in.node;
  tmp->_nodes()[1] = out.node;

  // fill-up the inputs and outputs
  tmp->_input_tensors()[0] = in;
  tmp->_output_tensors()[0] = out;

  // return the created pointer
  return std::move(tmp);
}

// static
command_ptr_t command_t::create_apply(
  command_id_t id,
  ud_impl_id_t fun_id,
  bool is_gpu,
  const std::vector<command_param_t> &params,
  const std::vector<tid_node_id_t> &in,
  const std::vector<tid_node_id_t> &out)
{
  // make sure all of them are at the same node
  assert(std::all_of(
    out.begin(),
    out.end(),
    [&](const tid_node_id_t &o) { return o.node == out[0].node; }));
  assert(std::all_of(
    in.begin(),
    in.end(),
    [&](const tid_node_id_t &i) { return i.node == out[0].node; }));

  // create the output
  auto tmp = allocate_command(_num_bytes(params.size(), 1, in.size(), out.size()));

  // set the id type and function
  tmp->id = id;
  tmp->type = APPLY;
  tmp->fun_id = fun_id;
  tmp->nfo.is_gpu = is_gpu;
  tmp->_num_parameters = params.size();
  tmp->_num_nodes = 1;
  tmp->_num_inputs = in.size();
  tmp->_num_outputs = out.size();

  // setup the offsets
  tmp->_setup_offsets();

  // fill-up the nodes - APPLY is local to a node and has to have at least one output tensor
  tmp->_nodes()[0] = out[0].node;

  // fill-up the parameters
  for(size_t idx = 0; idx < params.size(); ++idx) {
    tmp->_parameters()[idx] = params[idx];
  }

  // fill-up the inputs
  for(size_t idx = 0; idx < in.size(); ++idx) {
    tmp->_input_tensors()[idx] = in[idx];
  }

  // fill-up the outputs
  for(size_t idx = 0; idx < out.size(); ++idx) {
    tmp->_output_tensors()[idx] = out[idx];
  }

  // return the created pointer
  return std::move(tmp);
}

// static
command_ptr_t command_t::create_touch(
  command_id_t id,
  ud_impl_id_t fun_id,
  bool is_gpu,
  int which_input,
  int num_touches,
  const std::vector<command_param_t> &params_without_compact_and_which,
  const std::vector<tid_node_id_t> &in,
  const tid_node_id_t &out)
{
  std::vector <command_param_t> params(2 + params_without_compact_and_which.size());
  params[0].b = false;       // touches are not commpacts
  params[1].u = which_input; // the `which_input` serves as an identifier for the
                             // is the input tensor, most likely
  std::copy(
    params_without_compact_and_which.begin(),
    params_without_compact_and_which.end(),
    params.begin() + 2);

  // now that we have fixed the parameters, this is just an apply with a type of
  // TOUCH and nfo num_writes set
  command_ptr_t tmp = create_apply(id, fun_id, is_gpu, params, in, {out});

  tmp->type = TOUCH;
  tmp->nfo.num_writes = num_touches;

  return std::move(tmp);
}

// static
command_ptr_t command_t::create_compact(
  command_id_t id,
  ud_impl_id_t fun_id,
  bool is_gpu,
  int which_input,
  const std::vector<command_param_t> &params_without_compact_and_which,
  const tid_node_id_t &in,
  const tid_node_id_t &out)
{
  std::vector <command_param_t> params(2 + params_without_compact_and_which.size());
  params[0].b = true;        // yes, this is a compact
  params[1].i = which_input; // the `which_input` tells the kernel which partition of
                             // the output this kernel is writing to

  std::copy(
    params_without_compact_and_which.begin(),
    params_without_compact_and_which.end(),
    params.begin() + 2);

  // Now there is no compact type, this is really just an apply.
  command_ptr_t tmp = create_apply(id, fun_id, is_gpu, params, {in}, {out});

  return std::move(tmp);
}

// static
command_ptr_t command_t::create_broadcast(
  command_id_t id,
  tid_node_id_t in,
  const std::vector<tid_node_id_t> &out)
{
  // make sure we are talking about the same tensor and all the them are not the input node
  assert(std::all_of(out.begin(), out.end(), [&](const tid_node_id_t &o) { return o.tid == in.tid && o.node != in.node; }));

  // make sure all of the outputs are unique
  ////auto it = std::unique(out.begin(), out.end());
  ////assert((it == out.end()));

  // create  the output
  auto tmp = allocate_command(_num_bytes(0, 1u + out.size(), 1, out.size()));

  // set the id type and function
  tmp->id = id;
  tmp->type = MOVE;
  tmp->fun_id = {-1, -1};
  tmp->_num_parameters = 0;
  tmp->_num_nodes = 1u + out.size();
  tmp->_num_inputs = 1;
  tmp->_num_outputs = out.size();

  // setup the offsets
  tmp->_setup_offsets();

  // fill-up the nodes, broadcast goes from the input node to all the other nodes, not duplicates are allowed
  tmp->_nodes()[0] = in.node;
  for(size_t idx = 0; idx < out.size(); ++idx) {
    tmp->_nodes()[1 + idx] = out[idx].node;
  }

  // fill-up the inputs
  tmp->_input_tensors()[0] = in;

  // fill-up the outputs
  for(size_t idx = 0; idx < out.size(); ++idx) {
    tmp->_output_tensors()[idx] = out[idx];
  }

  // return the created pointer
  return std::move(tmp);
}

// static
command_ptr_t command_t::create_reduce(
  command_id_t id,
  ud_impl_id_t fun_id,
  bool is_gpu,
  const std::vector<command_param_t> &params,
  const std::vector<tid_node_id_t> &in, const tid_node_id_t &out)
{
  // the nodes
  std::vector<node_id_t> nodes;
  nodes.reserve(1u + in.size());

  // the root is at the out node
  nodes.push_back(out.node);
  for(const auto &i : in) {
    // check if we already have this node
    if(std::find(nodes.begin(), nodes.end(), i.node) == nodes.end()) {
      nodes.push_back(i.node);
    }
  }

  // create the output
  auto tmp = allocate_command(_num_bytes(params.size(), nodes.size(), in.size(), 1));

  // set the id type and function
  tmp->id = id;
  tmp->type = REDUCE;
  tmp->fun_id = fun_id;
  tmp->nfo.is_gpu = is_gpu;
  tmp->_num_parameters = params.size();
  tmp->_num_inputs = in.size();
  tmp->_num_outputs = 1;
  tmp->_num_nodes = nodes.size();

  // setup the offsets
  tmp->_setup_offsets();

  // fill-up the parameters
  for(size_t idx = 0; idx < params.size(); ++idx) {
    tmp->_parameters()[idx] = params[idx];
  }

  // fill-up the nodes
  for(size_t idx = 0; idx < nodes.size(); ++idx) {
    tmp->_nodes()[idx] = nodes[idx];
  }

  // fill-up the inputs
  for(size_t idx = 0; idx < in.size(); ++idx) {
    tmp->_input_tensors()[idx] = in[idx];
  }

  // fill-up the outputs
  tmp->_output_tensors()[0] = out;

  // return the created pointer
  return std::move(tmp);
}

// static
command_ptr_t command_t::create_delete(
  command_id_t id,
  const std::vector<tid_node_id_t> &in)
{

  // make sure all of the inputs are on the same node
  assert(std::all_of(
    in.begin(),
    in.end(),
    [&](const tid_node_id_t &i) { return i.node == in[0].node; }));

  // create the output
  auto tmp = allocate_command(_num_bytes(0, 1, in.size(), 0));

  // set the id type and function
  tmp->id = id;
  tmp->type = DELETE;
  tmp->fun_id = {-1, -1};
  tmp->_num_parameters = 0;
  tmp->_num_inputs = in.size();
  tmp->_num_outputs = 0;
  tmp->_num_nodes = 1;

  // setup the offsets
  tmp->_setup_offsets();

  // set the node
  tmp->_nodes()[0] = in[0].node;

  // fill-up the inputs
  for(size_t idx = 0; idx < in.size(); ++idx) {
    tmp->_input_tensors()[idx] = in[idx];
  }

  // return the created pointer
  return std::move(tmp);
}

// crates a shutdown command
// static
command_ptr_t command_t::create_shutdown(node_id_t node)
{
  // allocate the memory
  auto tmp = allocate_command(_num_bytes(0, 1, 0, 0));

  // set the id type and function
  tmp->id = -1;
  tmp->type = SHUTDOWN;
  tmp->fun_id = {-1, -1};
  tmp->_num_parameters = 0;
  tmp->_num_inputs = 0;
  tmp->_num_outputs = 0;
  tmp->_num_nodes = 1;

  // setup the offsets
  tmp->_setup_offsets();

  // set the node
  tmp->_nodes()[0] = node;

  // return the created pointer
  return std::move(tmp);
}

} // bbts

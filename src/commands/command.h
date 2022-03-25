#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>
#include <assert.h>
#include <cstring>
#include "command_utils.h"
#include "../ud_functions/ud_function.h"
#include "../server/node_config.h"

namespace bbts {

// the impl_id of the operation, this is unique globally across all processes
using command_id_t = int32_t;

// pre-declare the command ptr type as well as a deleter for it
struct command_t;
struct command_deleter_t { void operator()(command_t* p) { delete[] ((char*) p); }};
using command_ptr_t = std::unique_ptr<bbts::command_t, command_deleter_t>;

// the commands we execute, they can be copied directly with a memcpy as they are layered out flat
struct command_t {

  enum op_type_t : int32_t {

    APPLY = 0,
    REDUCE = 1,
    MOVE = 2,
    DELETE = 3,
    SHUTDOWN = 4 // special command to shutdown the server
  };

  // specifies exactly what tensor on which node we refer to
  struct tid_node_id_t {
    tid_t tid;
    node_id_t node;
  };

  // the list of tensors
  using tensor_id_list_t = raw_vector_t<tid_node_id_t>;

  // the list of nodes
  using node_list_t = raw_vector_t<node_id_t>;

  // return the number of input tensors
  [[nodiscard]] size_t get_num_inputs() const { return _num_inputs; }

  // returns the input tensor
  [[nodiscard]] tid_node_id_t &get_input(int32_t idx) {
    return _input_tensors() [idx];
  }

  // return the input but constant
  [[nodiscard]] const tid_node_id_t &get_input(int32_t idx) const {
    return _input_tensors() [idx];
  }

  // returns all the inputs as vector
  [[nodiscard]] tensor_id_list_t get_inputs() const {
    return tensor_id_list_t { ._data = _input_tensors(), ._num_elements = get_num_inputs() };
  }

  // returns the output tensor
  [[nodiscard]] tid_node_id_t &get_output(int32_t idx) {
    return _output_tensors() [idx];
  }

  // return the output tensor but constant
  [[nodiscard]] const tid_node_id_t &get_output(int32_t idx) const {
    return _output_tensors() [idx];
  }

  // returns all the outputs as vector
  [[nodiscard]] tensor_id_list_t get_outputs() const {
    return tensor_id_list_t { ._data = _output_tensors(), ._num_elements = get_num_outputs() };
  }

  // return the number of output tensors
  [[nodiscard]] size_t get_num_outputs() const { return _num_outputs; }

  // is this a delete
  [[nodiscard]] bool is_delete() const { return type == op_type_t::DELETE; }

  // return the number of bytes
  [[nodiscard]] size_t num_bytes() const { return _num_bytes(_num_parameters, _num_nodes, _num_inputs, _num_outputs); }

  // is this a move
  [[nodiscard]] bool is_move() const { return type == op_type_t::MOVE && _num_outputs == 1; }

  // is this a broadcast
  [[nodiscard]] bool is_broadcast() const { return type == op_type_t::MOVE && _num_outputs != 1; }

  // is this an apply
  [[nodiscard]] bool is_apply() const { return type == op_type_t::APPLY; }

    // is this a reduce
  [[nodiscard]] bool is_reduce() const { return type == op_type_t::REDUCE; }

  // get all the nodes included in the reduce
  [[nodiscard]] node_list_t get_nodes() const {
    return { ._data = _nodes(), ._num_elements = _num_nodes };
  }

  [[nodiscard]] command_param_list_t get_parameters() {
    return { ._data = _parameters(), ._num_elements = _num_parameters };
  }

  [[nodiscard]] tid_node_id_t get_reduce_input(node_id_t _node_id) {

    // try to find an input for this node
    auto inputs = get_inputs();
    for(auto in : inputs) {
      if(in.node == _node_id) {
        return in;
      }
    }

    return {-1, -1};
  }

  // is this a local reduce operator
  [[nodiscard]] bool is_local_reduce(node_id_t _node_id) const {

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

  // is remote reduce
  [[nodiscard]] bool is_remote_reduce(node_id_t _node_id) const {

    // make sure it is actually a reduce
    if(type != op_type_t::REDUCE) {
      return false;
    }

    // if it is not local it is remote
    return !is_local_reduce(_node_id);
  }

  // check if command uses a particular node
  [[nodiscard]] bool uses_node(node_id_t target_node) const {

    // go and check every tensor if it is located on a node
    auto nodes = get_nodes();
    for(auto &node : nodes) {
      if(node == target_node) {
        return true;
      }
    }

    return false;
  }

  void print(std::stringstream &ss) {

    ss << "{.id : " << id << " ";
    ss << ".type : ";

    switch (type) {
      case MOVE : ss << (_num_outputs == 1 ? "MOVE " : "BROADCAST ") ; break;
      case APPLY : ss << "APPLY "; break;
      case DELETE : ss << "DELETE "; break;
      case REDUCE : ss << "REDUCE "; break;
      case SHUTDOWN :ss << "SHUTDOWN ";  break;
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

  // the root node is always first
  [[nodiscard]] node_id_t get_root_node() const {
    return get_nodes()[0];
  }

  // clone the command
  command_ptr_t clone() {

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
  static command_ptr_t allocate_command(size_t num_bytes) {

    // allocate the memory
    std::unique_ptr<char[]> p(new char[num_bytes]);

    auto pReleased = p.release();
    auto pDerived = (bbts::command_t *)(pReleased);
    auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(pDerived);

    // move the command
    return std::move(d);
  }

  static command_ptr_t create_move(command_id_t id, tid_node_id_t in, tid_node_id_t out) {

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

  static command_ptr_t create_apply(command_id_t id, ud_impl_id_t fun_id, bool is_gpu, const std::vector<command_param_t> &params,
                                    const std::vector<tid_node_id_t> &in, const std::vector<tid_node_id_t> &out) {

    // make sure all of them are at the same node
    assert(std::all_of(out.begin(), out.end(), [&](const tid_node_id_t &o) { return o.node == out[0].node; }));
    assert(std::all_of(in.begin(),  in.end(),  [&](const tid_node_id_t &i) { return i.node == out[0].node; }));

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

  static command_ptr_t create_broadcast(command_id_t id, tid_node_id_t in, const std::vector<tid_node_id_t> &out) {

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

  static command_ptr_t create_reduce(command_id_t id, ud_impl_id_t fun_id, bool is_gpu,
                                     const std::vector<command_param_t> &params,
                                     const std::vector<tid_node_id_t> &in, const tid_node_id_t &out) {

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

  static command_ptr_t create_delete(command_id_t id, const std::vector<tid_node_id_t> &in) {

    // make sure all of the inputs are on the same node
    assert(std::all_of(in.begin(), in.end(), [&](const tid_node_id_t &i) { return i.node == in[0].node; }));

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
  static command_ptr_t create_shutdown(node_id_t node) {

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

  // the impl_id of the operation
  command_id_t id;

  // the type of operation
  op_type_t type;

  // the function we want to execute
  ud_impl_id_t fun_id = {-1, -1};

  // additional information about the command
  union {

    // this is used by the MOVE and broadcast command to send over the number of bytes
    size_t num_bytes;

    // is this command using the gpu
    bool is_gpu = false;

  } nfo;

private:

  // the number of parameters
  uint16_t _num_parameters;

  // the number of nodes
  uint16_t _num_nodes;

  // the number of input tensors
  uint16_t _num_inputs;

  // the number of output tensors
  uint16_t _num_outputs;

  // where the parameters start
  uint16_t _params_offset;

  // the nodes involved
  uint16_t _node_offset;

  // the tensors input
  uint16_t _input_tensor_offset;

  // the output
  uint16_t _output_tensor_offset;

  // setup all the offsets
  void _setup_offsets() {

    // we start here
    auto s = (int8_t *) this;

    // calculate the pointers for parameters
    auto e = s + sizeof(command_t);
    _params_offset = (uint16_t) (e - s);

    // calculate were the nodes begin
    e = e + _num_parameters * sizeof(command_param_t);
    _node_offset = (uint16_t) (e - s);

    // calculate were the inputs begin
    e = e + _num_nodes * sizeof(node_id_t);
    _input_tensor_offset = (uint16_t) (e - s);

    // calculate were the outputs begin
    e = e + _num_inputs * sizeof(tid_node_id_t);
    _output_tensor_offset = (uint16_t) (e - s);
  }

  // these return the offsets to parameters
  inline command_param_t* _parameters() const { return ((command_param_t *) (((int8_t *) this) + _params_offset)); }
  inline node_id_t* _nodes() const { return ((node_id_t *) (((int8_t *) this) + _node_offset)); }
  inline tid_node_id_t* _input_tensors() const { return ((tid_node_id_t *) (((int8_t *) this) + _input_tensor_offset)); }
  inline tid_node_id_t* _output_tensors() const { return ((tid_node_id_t *) (((int8_t *) this) + _output_tensor_offset)); }

  // the number of bytes
  static size_t _num_bytes(size_t num_parameters,
                           size_t num_nodes,
                           size_t num_inputs,
                           size_t num_outputs) {

    return sizeof(bbts::command_t) + num_parameters * sizeof (command_param_t) +
           num_nodes * sizeof(node_id_t) + (num_inputs + num_outputs) * sizeof(tid_node_id_t);
  }
};

}

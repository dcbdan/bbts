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
    SHUTDOWN = 4,// special command to shutdown the server
    TOUCH = 5,
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

   // is this a touch
  [[nodiscard]] bool is_touch() const { return type == op_type_t::TOUCH; }

  // get all the nodes included in the reduce
  [[nodiscard]] node_list_t get_nodes() const {
    return { ._data = _nodes(), ._num_elements = _num_nodes };
  }

  [[nodiscard]] command_param_list_t get_parameters() {
    return { ._data = _parameters(), ._num_elements = _num_parameters };
  }

  [[nodiscard]] tid_node_id_t get_reduce_input(node_id_t _node_id) const;

  // is this a local reduce operator
  [[nodiscard]] bool is_local_reduce(node_id_t _node_id) const;

  // is remote reduce
  [[nodiscard]] bool is_remote_reduce(node_id_t _node_id) const;

  // check if command uses a particular node
  [[nodiscard]] bool uses_node(node_id_t target_node) const;

  void print(std::stringstream &ss);

  // the root node is always first
  [[nodiscard]] node_id_t get_root_node() const {
    return get_nodes()[0];
  }

  // clone the command
  command_ptr_t clone();

  // allocate the command
  static command_ptr_t allocate_command(size_t num_bytes);

  static command_ptr_t create_move(command_id_t id, tid_node_id_t in, tid_node_id_t out);

  static command_ptr_t create_apply(
    command_id_t id,
    ud_impl_id_t fun_id,
    bool is_gpu,
    const std::vector<command_param_t> &params,
    const std::vector<tid_node_id_t> &in,
    const std::vector<tid_node_id_t> &out);

  static command_ptr_t create_broadcast(
    command_id_t id,
    tid_node_id_t in,
    const std::vector<tid_node_id_t> &out);

  static command_ptr_t create_reduce(
    command_id_t id,
    ud_impl_id_t fun_id,
    bool is_gpu,
    const std::vector<command_param_t> &params,
    const std::vector<tid_node_id_t> &in, const tid_node_id_t &out);

  static command_ptr_t create_touch(
    command_id_t id,
    ud_impl_id_t fun_id,
    bool is_gpu,
    int which_input,
    int num_touches,
    const std::vector<command_param_t> &params_without_compact_and_which,
    const std::vector<tid_node_id_t> &in,
    const tid_node_id_t &out);

  static command_ptr_t create_compact(
    command_id_t id,
    ud_impl_id_t fun_id,
    bool is_gpu,
    int which_input,
    const std::vector<command_param_t> &params_without_compact_and_which,
    const tid_node_id_t &in,
    const tid_node_id_t &out);

  static command_ptr_t create_delete(
    command_id_t id,
    const std::vector<tid_node_id_t> &in);

  // crates a shutdown command
  static command_ptr_t create_shutdown(node_id_t node);

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

    // this is used by TOUCH commands to determine the total number of writes that will
    // have to happen to finish the tensor
    size_t num_writes;

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
  inline command_param_t* _parameters() const {
    return ((command_param_t *) (((int8_t *) this) + _params_offset));
  }

  inline node_id_t* _nodes() const {
    return ((node_id_t *) (((int8_t *) this) + _node_offset));
  }

  inline tid_node_id_t* _input_tensors() const {
    return ((tid_node_id_t *) (((int8_t *) this) + _input_tensor_offset));
  }

  inline tid_node_id_t* _output_tensors() const {
    return ((tid_node_id_t *) (((int8_t *) this) + _output_tensor_offset));
  }

  // the number of bytes
  static size_t _num_bytes(size_t num_parameters,
                           size_t num_nodes,
                           size_t num_inputs,
                           size_t num_outputs)
  {
    return sizeof(bbts::command_t) + num_parameters * sizeof (command_param_t) +
           num_nodes * sizeof(node_id_t) + (num_inputs + num_outputs) * sizeof(tid_node_id_t);
  }
};

}

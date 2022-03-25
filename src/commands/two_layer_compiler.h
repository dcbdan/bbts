#pragma once

#include "abstract_command.h"
#include "command.h"
#include "cost_model.h"
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <list>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bbts {

// this compiler compiles the graph by repeatingly splitting the
// DAG of commands into two layers and applying an optimizer on them
class two_layer_compiler {
public:
  two_layer_compiler(cost_model_ptr_t cost_model, size_t num_nodes)
      : cost_model(cost_model), num_nodes(num_nodes), _node_costs(num_nodes),
        _moved_tensors(num_nodes) {}

  struct node_cost_t {

    // we use this as an estimate of how much data was transfered by the node
    double transfer_cost = 0;

    // we use this as an estimate of how much was computed
    double compute_cost = 0;

    // we use this to
    double gpu_cost = 0;

    // we want to use this
    double gpu_transfer_cost = 0;
  };

  std::vector<bbts::command_ptr_t>
  compile(const std::vector<abstract_command_t> &commands,
          std::vector<std::unordered_set<tid_t>> &tensor_locations);

  std::unordered_set<tid_t>
  _create_delete_commands(
      const std::vector<abstract_command_t> &commands,
      std::vector<std::unordered_set<tid_t>> &tensor_locations,
      std::vector<bbts::command_ptr_t> &generated_cmds);


  void _create_moved_commands(std::vector<bbts::command_ptr_t> &generated_cmds,
                              const std::unordered_set<tid_t> &delted);

  void optimize(const std::vector<abstract_command_t> &commands,
                std::vector<bbts::command_ptr_t> &generated_cmds,
                const std::vector<std::list<uint32_t>> &first_layer,
                const std::vector<std::list<uint32_t>> &second_layer,
                std::vector<std::unordered_set<tid_t>> &tensor_locations);

  void apply_rule_0(const std::vector<abstract_command_t> &commands,
                    const std::vector<std::list<uint32_t>> &producers,
                    std::vector<std::unordered_set<tid_t>> &tensor_locations,
                    std::vector<bbts::command_ptr_t> &generated_cmds);


  void apply_rule_1(node_id_t node,
                    const std::vector<abstract_command_t> &commands,
                    const std::list<uint32_t> &consumer,
                    const std::vector<std::list<uint32_t>> &producers,
                    std::vector<std::unordered_set<tid_t>> &tensor_locations,
                    std::vector<bbts::command_ptr_t> &generated_cmds);

  void apply_rule_2(node_id_t consumer_node,
                    const std::vector<node_id_t> producer_nodes,
                    const std::vector<abstract_command_t> &commands,
                    const std::list<uint32_t> &consumer,
                    const std::vector<std::list<uint32_t>> &producers,
                    std::vector<std::unordered_set<tid_t>> &tensor_locations,
                    std::vector<bbts::command_ptr_t> &generated_cmds);

  // put everything at one node
  std::tuple<node_id_t, float>
  rule_1(const std::vector<abstract_command_t> &commands,
         const std::list<uint32_t> &consumer,
         const std::vector<std::list<uint32_t>> &producers,
         std::vector<std::unordered_set<tid_t>> &tensor_locations);


  // put the producers where they have the smallest
  // execution overhead then place the consumer
  std::tuple<float, node_id_t, std::vector<node_id_t>>
  rule_2(const std::vector<abstract_command_t> &commands,
         const std::list<uint32_t> &consumer,
         const std::vector<std::list<uint32_t>> &producers,
         std::vector<std::unordered_set<tid_t>> &tensor_locations);

  // revert the changes to tensor locations
  void revert(const std::vector<std::tuple<tid_t, node_id_t>> &rule_history,
              std::vector<std::unordered_set<tid_t>> &tensor_locations);

  // add the previously undone
  void apply(const std::vector<std::tuple<tid_t, node_id_t>> &rule_history,
             std::vector<std::unordered_set<tid_t>> &tensor_locations);

  node_id_t _find_node_to_fetch(
      tid_t id,
      const std::vector<std::unordered_set<tid_t>> &tensor_locations);

  // update the move op
  void _update_move(std::vector<bbts::command_ptr_t> &out_commands,
                    command_id_t cmd_id, node_id_t best_node);

  void
  generate_for_node(const std::list<uint32_t> &cmd,
                    const std::vector<abstract_command_t> &commands,
                    node_id_t node,
                    std::vector<std::unordered_set<tid_t>> &tensor_locations,
                    std::vector<bool> &gpu_assigment_best,
                    std::vector<bbts::command_ptr_t> &generated_cmds);

  void generate_or_update_move(
      tid_t tid, node_id_t node,
      std::vector<std::unordered_set<tid_t>> &tensor_locations,
      std::vector<bbts::command_ptr_t> &generated_cmds);

  void backtrace(std::vector<std::tuple<char, char>> trace, char best,
                 std::vector<bool> &gpu_assigment);

  // returns the {transfer_cost, cpu_cost, gpu_cost}
  std::tuple<float, float, float, float>
  calculate_cost(node_id_t node, const std::list<uint32_t> &cmd,
                 const std::vector<abstract_command_t> &commands,
                 std::vector<std::unordered_set<tid_t>> &tensor_locations,
                 std::vector<bool> &gpu_assigment);

  std::tuple<float, float, float, float>
  try_assign(node_id_t node, const std::list<uint32_t> &cmd,
             const std::vector<abstract_command_t> &commands,
             std::vector<std::unordered_set<tid_t>> &tensor_locations,
             std::vector<std::tuple<tid_t, node_id_t>> &history);

  void _update_present_tids(std::unordered_set<tid_t> &present_tids,
                            const std::vector<std::list<uint32_t>> &first_layer,
                            const std::vector<abstract_command_t> &commands);

  std::vector<std::list<uint32_t>>
  _get_layer(const std::vector<abstract_command_t> &commands,
             const std::unordered_set<tid_t> present_tids);

  void add_all_appendable(std::list<uint32_t> &op_list,
                          const std::vector<abstract_command_t> &commands);

  bool is_apendable(uint32_t &producer, uint32_t &consumer,
                    const std::vector<abstract_command_t> &commands);

private:
  // this maps the tensor to all the commands that consume it
  std::unordered_map<tid_t, std::vector<uint32_t>> _tensor_consumers;

  // maps the consumers commands to all the producer commands
  std::vector<std::vector<uint32_t>> _consumer_producer;

  // keeps track of all the move commands
  std::unordered_map<tid_t, uint64_t> _move_cmds;

  // all the tensors we eventually need to delete as they were duplicated by a
  // move
  std::vector<std::vector<tid_t>> _moved_tensors;

  // the max costs across all nodes
  node_cost_t max_cost;

  // transfer costs
  std::vector<node_cost_t> _node_costs;

  // maps the index in the command vector to number of inputs left
  std::vector<uint32_t> _inputs_left;

  // this tells us the cost of running the ud functions and transfering tensors
  cost_model_ptr_t cost_model;

  // the number of nodes in the cluster
  size_t num_nodes;
};

} // namespace bbts

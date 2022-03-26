#include "two_layer_compiler.h"

namespace bbts {

std::vector<command_ptr_t>
two_layer_compiler::compile(
  const std::vector<abstract_command_t> &commands,
  std::vector<std::unordered_set<tid_t>> &tensor_locations)
{
  std::vector<command_ptr_t> generated_cmds;

  // precompute the costs
  cost_model->precompute_costs(commands);

  // init the input counts
  _inputs_left.resize(commands.size());
  _tensor_consumers.clear();
  auto it = _inputs_left.begin();
  for (auto idx = 0; idx < commands.size(); ++idx) {
    const auto &c = commands[idx];
    *it = c.input_tids.size();
    it++;

    if (c.type == abstract_command_type_t::DELETE) {
      continue;
    }

    for (auto tid : c.input_tids) {
      _tensor_consumers[tid].push_back(idx);
    }
  }

  // flatten the tensor locations to avoid duplicates
  // and get all the tids that are present in our cluster
  std::unordered_set<tid_t> present_tids;
  for (const auto &tl : tensor_locations) {
    for (const auto tid : tl) {
      present_tids.insert(tid);
    }
  }

  {
    // go through all the commands that do not need an input
    // those are usually the ones that generate the some tensors
    std::vector<std::list<uint32_t>> _zero_layer;
    for (uint32_t idx = 0; idx < commands.size(); ++idx) {
      if (commands[idx].input_tids.empty()) {
        _zero_layer.push_back({idx});
      }
    }

    // update present tids
    _update_present_tids(present_tids, _zero_layer, commands);

    // run the optimizer here to assign this layer if necessary
    if (!_zero_layer.empty()) {
      optimize(commands, generated_cmds, _zero_layer, {}, tensor_locations);
    }
  }

  while (true) {

    // get the frist layer
    auto first_layer = _get_layer(commands, present_tids);

    // we are out of commands to optimize
    if (first_layer.empty()) {
      break;
    }

    // update present tids
    present_tids.clear();
    _update_present_tids(present_tids, first_layer, commands);

    // get the second layer
    auto second_layer = _get_layer(commands, present_tids);

    // run the optimier to find the optimal placment for the commmands
    optimize(commands, generated_cmds, first_layer, second_layer,
             tensor_locations);

    // update present tids
    present_tids.clear();
    _update_present_tids(present_tids, second_layer, commands);
  }

  // add the deleted commands
  auto deleted =
      _create_delete_commands(commands, tensor_locations, generated_cmds);

  // generate the commands to remove the moved tensors
  _create_moved_commands(generated_cmds, deleted);

  // return the generated commands
  return std::move(generated_cmds);
}

std::unordered_set<tid_t>
two_layer_compiler::_create_delete_commands(
    const std::vector<abstract_command_t> &commands,
    std::vector<std::unordered_set<tid_t>> &tensor_locations,
    std::vector<command_ptr_t> &generated_cmds)
{
  std::unordered_set<tid_t> ret;
  for (auto &c : commands) {

    // skip if not delete
    if (c.type != abstract_command_type_t::DELETE) {
      continue;
    }

    // keep track of what we have deleted
    for (auto &in : c.input_tids) {
      ret.insert(in);
    }

    // find the tids on each node
    for (node_id_t node = 0; node < num_nodes; ++node) {

      // check if there is something
      std::vector<command_t::tid_node_id_t> inputs;
      for (auto &in : c.input_tids) {

        // create the command and remove
        auto it = tensor_locations[node].find(in);
        if (it != tensor_locations[node].end()) {
          inputs.push_back(command_t::tid_node_id_t{.tid = in, .node = node});
          tensor_locations[node].erase(it);
        }
      }

      if (!inputs.empty()) {

        // make the apply
        auto cmd_id = generated_cmds.size();
        auto cmd = command_t::create_delete(cmd_id, inputs);

        // store the command
        generated_cmds.push_back(std::move(cmd));
      }
    }
  }

  return std::move(ret);
}

void two_layer_compiler::_create_moved_commands(
  std::vector<command_ptr_t> &generated_cmds,
  const std::unordered_set<tid_t> &delted)
{
  for (node_id_t node = 0; node < num_nodes; ++node) {
    std::vector<command_t::tid_node_id_t> inputs;
    for (auto &in : _moved_tensors[node]) {
      if (delted.find(in) != delted.end()) {
        continue;
      }
      inputs.push_back(command_t::tid_node_id_t{.tid = in, .node = node});
    }

    // check if there is something
    if (!inputs.empty()) {

      // make the apply
      auto cmd_id = generated_cmds.size();
      auto cmd = command_t::create_delete(cmd_id, inputs);

      // store the command
      generated_cmds.push_back(std::move(cmd));
    }
    _moved_tensors[node].clear();
  }
}

void two_layer_compiler::optimize(
  const std::vector<abstract_command_t> &commands,
  std::vector<command_ptr_t> &generated_cmds,
  const std::vector<std::list<uint32_t>> &first_layer,
  const std::vector<std::list<uint32_t>> &second_layer,
  std::vector<std::unordered_set<tid_t>> &tensor_locations)
{
  // just make sure the we actually have something to optimize
  if (first_layer.empty() && second_layer.empty()) {
    return;
  }

  // if have to only plan for the first layer apply rule 0 and exit
  if (second_layer.empty()) {
    apply_rule_0(commands, first_layer, tensor_locations, generated_cmds);
    return;
  }

  // index the first layer so that we know what command creates what tensor
  std::unordered_map<tid_t, uint32_t> first_layer_idx;
  for (uint32_t idx = 0; idx < first_layer.size(); ++idx) {
    for (auto out : commands[first_layer[idx].back()].output_tids) {
      first_layer_idx[out] = idx;
    }
  }

  // go through the second layer and apply for each command list the rule 1
  // and rule 2 then calculate the cost and pick the one with the smallest
  // cost
  std::vector<std::list<uint32_t>> producers;
  std::unordered_set<int32_t> processed_producer;
  for (const auto &consumer : second_layer) {

    producers.clear();
    for (auto in : commands[consumer.front()].input_tids) {
      auto idx = first_layer_idx[in];
      if (processed_producer.find(idx) == processed_producer.end()) {
        processed_producer.insert(idx);
        producers.emplace_back(first_layer[idx]);
      }
    }

    // run rule 1
    auto [r1_cost, r1_node] =
        rule_1(commands, consumer, producers, tensor_locations);

    // run rule 2
    auto [r2_cost, r2_consumer, r2_producers] =
        rule_2(commands, consumer, producers, tensor_locations);

    // apply with the smallest cost
    if (r1_cost < r2_cost) {
      apply_rule_1(r1_node, commands, consumer, producers, tensor_locations,
                   generated_cmds);
    } else {
      apply_rule_2(r2_consumer, r2_producers, commands, consumer, producers,
                   tensor_locations, generated_cmds);
    }
  }

  // do we have producers we have not assigned
  producers.clear();
  for (auto idx = 0; idx < first_layer.size(); idx++) {
    if (processed_producer.find(idx) == processed_producer.end()) {
      producers.emplace_back(first_layer[idx]);
    }
  }

  // if we do apply rule 0
  if (!producers.empty()) {
    apply_rule_0(commands, producers, tensor_locations, generated_cmds);
  }
}

void two_layer_compiler::apply_rule_0(
  const std::vector<abstract_command_t> &commands,
  const std::vector<std::list<uint32_t>> &producers,
  std::vector<std::unordered_set<tid_t>> &tensor_locations,
  std::vector<command_ptr_t> &generated_cmds)
{
  // go through all the commands
  node_cost_t added_cost;
  std::vector<bool> gpu_assigment_tmp; // true if the command is on the kernel
  std::vector<bool> gpu_assigment_best; // false if it is not
  for (auto &cmd : producers) {

    node_id_t best_node = 0;
    float best_overhead = std::numeric_limits<float>::max();

    // calculate the cost of running the command on each node
    for (auto node = 0; node < num_nodes; node++) {

      // cost calculation
      auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] = calculate_cost(
          node, cmd, commands, tensor_locations, gpu_assigment_tmp);

      // get the actual overhead (we want these to be balanced)
      auto transfer_overhead =
          std::max(_node_costs[node].transfer_cost + transfer_cost,
                   max_cost.transfer_cost);
      auto cpu_overhead = std::max(_node_costs[node].compute_cost + cpu_cost,
                                   max_cost.compute_cost);
      auto gpu_overhead =
          std::max(_node_costs[node].gpu_cost + gpu_cost, max_cost.gpu_cost);
      auto gpu_transfer_overhead =
          std::max(_node_costs[node].gpu_transfer_cost + gpu_transfer,
                   max_cost.gpu_transfer_cost);

      // check if this is the best node we can assign this to
      if (best_overhead > (transfer_overhead + cpu_overhead + gpu_overhead +
                           gpu_transfer_overhead))
      {
        best_node = node;
        best_overhead = transfer_overhead + cpu_overhead + gpu_overhead +
                        gpu_transfer_overhead;

        added_cost.transfer_cost = transfer_cost;
        added_cost.compute_cost = cpu_cost;
        added_cost.gpu_cost = gpu_cost;
        added_cost.gpu_transfer_cost = gpu_transfer;

        std::swap(gpu_assigment_best, gpu_assigment_tmp);
      }
    }

    // pick the one with the smallest cost, and generate the commands
    generate_for_node(cmd, commands, best_node, tensor_locations,
                      gpu_assigment_best, generated_cmds);

    // update the costs
    _node_costs[best_node].transfer_cost += added_cost.transfer_cost;
    _node_costs[best_node].compute_cost += added_cost.compute_cost;
    _node_costs[best_node].gpu_cost += added_cost.gpu_cost;
    _node_costs[best_node].gpu_transfer_cost += added_cost.gpu_transfer_cost;
  }
}

void two_layer_compiler::apply_rule_1(
  node_id_t node,
  const std::vector<abstract_command_t> &commands,
  const std::list<uint32_t> &consumer,
  const std::vector<std::list<uint32_t>> &producers,
  std::vector<std::unordered_set<tid_t>> &tensor_locations,
  std::vector<command_ptr_t> &generated_cmds)
{
  std::vector<bool> gpu_assigment;
  for (auto &p : producers) {
    gpu_assigment.clear();
    auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] =
        calculate_cost(node, p, commands, tensor_locations, gpu_assigment);

    // update the costs
    _node_costs[node].transfer_cost += transfer_cost;
    _node_costs[node].compute_cost += cpu_cost;
    _node_costs[node].gpu_cost += gpu_cost;
    _node_costs[node].gpu_transfer_cost += gpu_transfer;

    // generate the commands
    generate_for_node(p, commands, node, tensor_locations, gpu_assigment,
                      generated_cmds);
  }

  // assign the consumer
  gpu_assigment.clear();
  auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] = calculate_cost(
      node, consumer, commands, tensor_locations, gpu_assigment);

  // generate the commands
  generate_for_node(consumer, commands, node, tensor_locations, gpu_assigment,
                    generated_cmds);

  // update the costs
  _node_costs[node].transfer_cost += transfer_cost;
  _node_costs[node].compute_cost += cpu_cost;
  _node_costs[node].gpu_cost += gpu_cost;
  _node_costs[node].gpu_transfer_cost += gpu_transfer;
}

void two_layer_compiler::apply_rule_2(
  node_id_t consumer_node,
  const std::vector<node_id_t> producer_nodes,
  const std::vector<abstract_command_t> &commands,
  const std::list<uint32_t> &consumer,
  const std::vector<std::list<uint32_t>> &producers,
  std::vector<std::unordered_set<tid_t>> &tensor_locations,
  std::vector<command_ptr_t> &generated_cmds)
{
  std::vector<bool> gpu_assigment;
  for (auto idx = 0; idx < producers.size(); ++idx) {
    auto node = producer_nodes[idx];
    gpu_assigment.clear();
    auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] = calculate_cost(
        node, producers[idx], commands, tensor_locations, gpu_assigment);

    // update the costs
    _node_costs[node].transfer_cost += transfer_cost;
    _node_costs[node].compute_cost += cpu_cost;
    _node_costs[node].gpu_cost += gpu_cost;
    _node_costs[node].gpu_transfer_cost += gpu_transfer;

    // generate the commands
    generate_for_node(producers[idx], commands, node, tensor_locations,
                      gpu_assigment, generated_cmds);
  }

  // assign the consumer
  gpu_assigment.clear();
  auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] = calculate_cost(
      consumer_node, consumer, commands, tensor_locations, gpu_assigment);

  // update the costs
  _node_costs[consumer_node].transfer_cost += transfer_cost;
  _node_costs[consumer_node].compute_cost += cpu_cost;
  _node_costs[consumer_node].gpu_cost += gpu_cost;
  _node_costs[consumer_node].gpu_transfer_cost += gpu_transfer;

  // generate the commands
  generate_for_node(consumer, commands, consumer_node, tensor_locations,
                    gpu_assigment, generated_cmds);
}

// put everything at one node
std::tuple<node_id_t, float>
two_layer_compiler::rule_1(
  const std::vector<abstract_command_t> &commands,
  const std::list<uint32_t> &consumer,
  const std::vector<std::list<uint32_t>> &producers,
  std::vector<std::unordered_set<tid_t>> &tensor_locations)
{
  std::vector<std::tuple<tid_t, node_id_t>> tmp_history;
  float best_overhead = std::numeric_limits<float>::infinity();
  node_id_t best_node = 0;

  for (node_id_t node = 0; node < num_nodes; ++node) {

    tmp_history.clear();
    node_cost_t cost_history{};

    float assigment_overhead = 0;
    for (auto &p : producers) {
      auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] =
          try_assign(node, p, commands, tensor_locations, tmp_history);

      // get the actual overhead (we want these to be balanced)
      auto transfer_overhead =
          std::max(_node_costs[node].transfer_cost + transfer_cost,
                   max_cost.transfer_cost);
      auto cpu_overhead = std::max(_node_costs[node].compute_cost + cpu_cost,
                                   max_cost.compute_cost);
      auto gpu_overhead =
          std::max(_node_costs[node].gpu_cost + gpu_cost, max_cost.gpu_cost);
      auto gpu_transfer_overhead =
          std::max(_node_costs[node].gpu_transfer_cost + gpu_transfer,
                   max_cost.gpu_transfer_cost);

      // update the costs
      _node_costs[node].transfer_cost += transfer_cost;
      _node_costs[node].compute_cost += cpu_cost;
      _node_costs[node].gpu_cost += gpu_cost;
      _node_costs[node].gpu_transfer_cost += gpu_transfer;

      // store the update
      cost_history.transfer_cost += transfer_cost;
      cost_history.compute_cost += cpu_cost;
      cost_history.gpu_cost += gpu_cost;
      cost_history.gpu_transfer_cost += gpu_transfer;

      // add the overhead
      assigment_overhead += (transfer_overhead + cpu_overhead + gpu_overhead +
                             gpu_transfer_overhead);
    }

    auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] =
        try_assign(node, consumer, commands, tensor_locations, tmp_history);

    // get the actual overhead (we want these to be balanced)
    auto transfer_overhead =
        std::max(_node_costs[node].transfer_cost + transfer_cost,
                 max_cost.transfer_cost);
    auto cpu_overhead = std::max(_node_costs[node].compute_cost + cpu_cost,
                                 max_cost.compute_cost);
    auto gpu_overhead =
        std::max(_node_costs[node].gpu_cost + gpu_cost, max_cost.gpu_cost);
    auto gpu_transfer_overhead =
        std::max(_node_costs[node].gpu_transfer_cost + gpu_transfer,
                 max_cost.gpu_transfer_cost);

    assigment_overhead += (transfer_overhead + cpu_overhead + gpu_overhead +
                           gpu_transfer_overhead);

    revert(tmp_history, tensor_locations);
    if (best_overhead > assigment_overhead) {

      // we got something do the bookeeping
      best_node = node;
      best_overhead = assigment_overhead;
    }

    // revert the costs
    _node_costs[node].transfer_cost -= cost_history.transfer_cost;
    _node_costs[node].compute_cost -= cost_history.compute_cost;
    _node_costs[node].gpu_cost -= cost_history.gpu_cost;
    _node_costs[node].gpu_transfer_cost -= cost_history.gpu_transfer_cost;
  }

  return {best_overhead, best_node};
}

// put the producers where they have the smallest
// execution overhead then place the consumer
std::tuple<float, node_id_t, std::vector<node_id_t>>
two_layer_compiler::rule_2(
  const std::vector<abstract_command_t> &commands,
  const std::list<uint32_t> &consumer,
  const std::vector<std::list<uint32_t>> &producers,
  std::vector<std::unordered_set<tid_t>> &tensor_locations)
{
  std::vector<node_cost_t> cost_history(num_nodes);
  for (node_id_t node = 0; node < num_nodes; ++node) {
    cost_history[node] = _node_costs[node];
  }

  std::vector<std::tuple<tid_t, node_id_t>> assigment_history;

  float planning_overhead = 0.0f;
  std::vector<node_id_t> producer_assigments;
  producer_assigments.reserve(producers.size());
  std::vector<std::tuple<tid_t, node_id_t>> tmp_history;
  for (auto &p : producers) {

    node_id_t best_node = 0;
    float best_overhead = std::numeric_limits<float>::infinity();
    for (node_id_t node = 0; node < num_nodes; ++node) {

      auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] =
          try_assign(node, p, commands, tensor_locations, tmp_history);

      // get the actual overhead (we want these to be balanced)
      auto transfer_overhead =
          std::max(_node_costs[node].transfer_cost + transfer_cost,
                   max_cost.transfer_cost);
      auto cpu_overhead = std::max(_node_costs[node].compute_cost + cpu_cost,
                                   max_cost.compute_cost);
      auto gpu_overhead =
          std::max(_node_costs[node].gpu_cost + gpu_cost, max_cost.gpu_cost);
      auto gpu_transfer_overhead =
          std::max(_node_costs[node].gpu_transfer_cost + gpu_transfer,
                   max_cost.gpu_transfer_cost);

      auto total_overhead = (transfer_overhead + cpu_overhead + gpu_overhead +
                             gpu_transfer_overhead);
      if (total_overhead < best_overhead) {
        best_node = node;
        best_overhead = total_overhead;
      }

      // revert the changes
      revert(tmp_history, tensor_locations);
      tmp_history.clear();
    }

    // assing to this node
    producer_assigments.push_back(best_node);
    auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] = try_assign(
        best_node, p, commands, tensor_locations, assigment_history);
    planning_overhead += best_overhead;

    // update the costs
    _node_costs[best_node].transfer_cost += transfer_cost;
    _node_costs[best_node].compute_cost += cpu_cost;
    _node_costs[best_node].gpu_cost += gpu_cost;
    _node_costs[best_node].gpu_transfer_cost += gpu_transfer;
  }

  node_id_t best_node = 0;
  float best_overhead = std::numeric_limits<float>::infinity();
  for (node_id_t node = 0; node < num_nodes; ++node) {

    auto [transfer_cost, cpu_cost, gpu_cost, gpu_transfer] =
        try_assign(node, consumer, commands, tensor_locations, tmp_history);

    // get the actual overhead (we want these to be balanced)
    auto transfer_overhead =
        std::max(_node_costs[node].transfer_cost + transfer_cost,
                 max_cost.transfer_cost);
    auto cpu_overhead = std::max(_node_costs[node].compute_cost + cpu_cost,
                                 max_cost.compute_cost);
    auto gpu_overhead =
        std::max(_node_costs[node].gpu_cost + gpu_cost, max_cost.gpu_cost);
    auto gpu_transfer_overhead =
        std::max(_node_costs[node].gpu_transfer_cost + gpu_transfer,
                 max_cost.gpu_transfer_cost);

    auto total_overhead = (transfer_overhead + cpu_overhead + gpu_overhead +
                           gpu_transfer_overhead);
    if (total_overhead < best_overhead) {
      best_node = node;
      best_overhead = total_overhead;
    }

    // revert the changes
    revert(tmp_history, tensor_locations);
    tmp_history.clear();
  }
  planning_overhead += best_overhead;

  // revert the costs as we still have not commited to this rule
  for (node_id_t node = 0; node < num_nodes; ++node) {
    _node_costs[node].transfer_cost = cost_history[node].transfer_cost;
    _node_costs[node].compute_cost = cost_history[node].compute_cost;
    _node_costs[node].gpu_cost = cost_history[node].gpu_cost;
    _node_costs[node].gpu_transfer_cost =
        cost_history[node].gpu_transfer_cost;
  }
  revert(assigment_history, tensor_locations);

  // return the result
  return {planning_overhead, best_node, std::move(producer_assigments)};
}

// revert the changes to tensor locations
void two_layer_compiler::revert(
  const std::vector<std::tuple<tid_t, node_id_t>> &rule_history,
  std::vector<std::unordered_set<tid_t>> &tensor_locations)
{
  // erase the changes
  for (auto h : rule_history) {
    auto [tensor, node] = h;
    tensor_locations[node].erase(tensor);
  }
}

// add the previously undone
void two_layer_compiler::apply(
  const std::vector<std::tuple<tid_t, node_id_t>> &rule_history,
  std::vector<std::unordered_set<tid_t>> &tensor_locations)
{
  // add the changes
  for (auto h : rule_history) {
    auto [tensor, node] = h;
    tensor_locations[node].insert(tensor);
  }
}

node_id_t two_layer_compiler::_find_node_to_fetch(
  tid_t id,
  const std::vector<std::unordered_set<tid_t>> &tensor_locations)
{
  // init to find the best
  float currBestCost = std::numeric_limits<float>::max();
  node_id_t bestNode = -1;

  // check every node
  for (tid_t node = 0; node < num_nodes; ++node) {

    // check if the tensor is on this node
    auto it = tensor_locations[node].find(id);
    if (it != tensor_locations[node].end()) {

      // is this the best option
      auto c = _node_costs[node].transfer_cost;
      if (c < currBestCost) {
        currBestCost = c;
        bestNode = node;
      }
    }
  }
  // return the node
  return bestNode;
}

// update the move op
void two_layer_compiler::_update_move(
  std::vector<command_ptr_t> &out_commands,
  command_id_t cmd_id,
  node_id_t best_node)
{
  // store the previous command
  auto cmd = std::move(out_commands[cmd_id]);

  // copy the previous
  std::vector<command_t::tid_node_id_t> out;
  out.reserve(cmd->get_num_outputs() + 1);
  for (int32_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {
    out.push_back(cmd->get_output(idx));
  }
  out.push_back(command_t::tid_node_id_t{.tid = cmd->get_input(0).tid,
                                         .node = best_node});

  // store the command
  out_commands[cmd_id] =
      command_t::create_broadcast(cmd_id, cmd->get_input(0), out);
}

void
two_layer_compiler::generate_for_node(
  const std::list<uint32_t> &cmd,
  const std::vector<abstract_command_t> &commands,
  node_id_t node,
  std::vector<std::unordered_set<tid_t>> &tensor_locations,
  std::vector<bool> &gpu_assigment_best,
  std::vector<command_ptr_t> &generated_cmds)
{
  // if the first command is an apply we need to create a bunch of move
  // commands for the tensor that are not present otherwise the reduce takes
  // care of that by itself
  auto root_cmd = commands[cmd.front()];
  if (root_cmd.type == abstract_command_type_t::APPLY) {
    auto &present_tensors = tensor_locations[node];
    for (auto in : root_cmd.input_tids) {
      if (present_tensors.find(in) == present_tensors.end()) {
        generate_or_update_move(in, node, tensor_locations, generated_cmds);
      }
    }
  }

  auto is_gpu = gpu_assigment_best.begin();
  for (auto cmd_idx : cmd) {

    auto &c = commands[cmd_idx];
    command_ptr_t gen;
    if (c.type == abstract_command_type_t::APPLY) {

      auto cmd_id = generated_cmds.size();

      // init the inputs
      std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
      for (int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
        inputs[idx] =
            command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = node};
      }

      // init the outputs
      std::vector<command_t::tid_node_id_t> outputs(c.output_tids.size());
      for (int32_t idx = 0; idx < c.output_tids.size(); ++idx) {

        // store the output location
        outputs[idx] =
            command_t::tid_node_id_t{.tid = c.output_tids[idx], .node = node};

        // mark that we are creating a tensor here
        tensor_locations[node].insert(c.output_tids[idx]);
      }

      // init the parameters
      std::vector<command_param_t> params(c.params.size());
      for (int32_t idx = 0; idx < c.params.size(); ++idx) {
        params[idx] = c.params[idx];
      }

      // gives us the info abou
      auto ud_info = cost_model->get_ud_choice(c.ud_id);

      // create the apply
      gen = command_t::create_apply(
          cmd_id, *is_gpu ? ud_info.gpu->impl_id : ud_info.cpu->impl_id,
          *is_gpu, params, inputs, outputs);

    } else {

      // init the inputs
      std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
      int32_t num_to_fetch = 0;
      for (int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
        node_id_t tid_node = node;
        if (tensor_locations[node].find(c.input_tids[idx]) ==
            tensor_locations[node].end()) {
          tid_node = _find_node_to_fetch(c.input_tids[idx], tensor_locations);
          num_to_fetch++;
        }
        inputs[idx] = command_t::tid_node_id_t{.tid = c.input_tids[idx],
                                               .node = tid_node};
      }

      // check if each tensor is not on this node
      // if it is we need to insert at least one move so that REDUCE will work
      assert(!inputs.empty());
      if (num_to_fetch == inputs.size()) {
        inputs.begin()->node = node;
        generate_or_update_move(inputs.begin()->tid, node, tensor_locations,
                                generated_cmds);
      }

      // init the outputs
      auto output =
          command_t::tid_node_id_t{.tid = c.output_tids[0], .node = node};

      // mark that we are creating a tensor here
      tensor_locations[node].insert(c.output_tids[0]);

      // init the parameters
      std::vector<command_param_t> params(c.params.size());
      for (int32_t idx = 0; idx < c.params.size(); ++idx) {
        params[idx] = c.params[idx];
      }

      // gives us the info abou
      auto ud_info = cost_model->get_ud_choice(c.ud_id);

      // create the reduce
      auto cmd_id = generated_cmds.size();
      gen = command_t::create_reduce(
          cmd_id, *is_gpu ? ud_info.gpu->impl_id : ud_info.cpu->impl_id,
          *is_gpu, params, inputs, output);
    }

    is_gpu++;
    generated_cmds.push_back(std::move(gen));
  }
}

void two_layer_compiler::generate_or_update_move(
  tid_t tid, node_id_t node,
  std::vector<std::unordered_set<tid_t>> &tensor_locations,
  std::vector<command_ptr_t> &generated_cmds)
{
  // Either create a new move command or piggyback a previous move
  // with a broadcast

  // ok we are moving this one
  _moved_tensors[node].push_back(tid);

  // do we have an existing command to do the move
  auto it = _move_cmds.find(tid);
  if (it == _move_cmds.end()) {

    // find the node to fetch from
    auto from_node = _find_node_to_fetch(tid, tensor_locations);

    // create the move command to this node
    auto cur_cmd = generated_cmds.size();
    _move_cmds[tid] = cur_cmd;
    auto cmd = command_t::create_move(
        cur_cmd, command_t::tid_node_id_t{.tid = tid, .node = from_node},
        command_t::tid_node_id_t{.tid = tid, .node = node});
    generated_cmds.push_back(std::move(cmd));

    // mark the location
    tensor_locations[node].insert(tid);

  } else {

    // update the move command
    auto cmd_id = it->second;
    _update_move(generated_cmds, cmd_id, node);

    // mark the location
    tensor_locations[node].insert(tid);
  }
}

void two_layer_compiler::backtrace(
  std::vector<std::tuple<char, char>> trace, char best,
  std::vector<bool> &gpu_assigment)
{
  int64_t cur = trace.size() - 1;
  gpu_assigment.resize(trace.size());

  while (cur >= 0) {

    // assign it
    if (best == -1) {
      throw std::runtime_error(
          "No kernel assigned in the backtrace! How did this happen?");
    }
    gpu_assigment[cur] = (bool)best;

    // is it on the CPU
    if (best == 0 && std::get<0>(trace[cur]) == 0) {
      best = 1;
    }
    // is it on the GPU
    else if (best == 1 && std::get<1>(trace[cur]) == 0) {
      best = 0;
    }

    // onto the next one
    cur--;
  }
}

// returns the {transfer_cost, cpu_cost, gpu_cost}
std::tuple<float, float, float, float>
two_layer_compiler::calculate_cost(
  node_id_t node, const std::list<uint32_t> &cmd,
  const std::vector<abstract_command_t> &commands,
  std::vector<std::unordered_set<tid_t>> &tensor_locations,
  std::vector<bool> &gpu_assigment)
{
  float transfer_cost = 0.0f;
  // the tensors present on the node
  auto &present_tensors = tensor_locations[node];

  // go through all the required inputs and sum the costs
  for (auto in : commands[cmd.front()].input_tids) {
    auto cst = cost_model->get_transfer_cost(in);
    if (present_tensors.find(in) == present_tensors.end()) {
      transfer_cost += cst.network_transfer_cost;
    }
  }

  // check if the tensor is on the node
  float cpu_cost[2] = {
      0.0f, std::numeric_limits<float>::infinity()}; // 0 off the GPU
  float gpu_cost[2] = {
      0.0f, std::numeric_limits<float>::infinity()}; // 1 on the GPU
  float gpu_transfer[2] = {
      0.0f, std::numeric_limits<float>::infinity()}; // 1 on the GPU

  std::vector<std::tuple<char, char>> choices;
  choices.reserve(cmd.size());

  for (auto &c : cmd) {

    // calculate to run the kernels
    auto cst = cost_model->get_execution_cost(c);

    // calculate the cost to transfer the input to the GPU
    float gpu_transfer_cost = 0.0f;
    for (auto in : commands[c].input_tids) {
      auto tcst = cost_model->get_transfer_cost(in);
      if (present_tensors.find(in) == present_tensors.end()) {
        gpu_transfer_cost += tcst.gpu_transfer_cost;
      }
    }

    // the total cost
    auto on_cpu_total = cpu_cost[0] + gpu_cost[0] + gpu_transfer[0];
    auto on_gpu_total = cpu_cost[1] + gpu_cost[1] + gpu_transfer[1];

    choices.push_back({});
    if (cst.is_gpu()) {
      if ((on_gpu_total + cst.gpu) <
          (on_cpu_total + gpu_transfer_cost + cst.gpu)) {
        std::get<1>(choices.back()) = 1;
        gpu_cost[1] += cst.gpu;
      } else {
        cpu_cost[1] = cpu_cost[0];
        gpu_transfer[1] = gpu_transfer[0] + gpu_transfer_cost;
        gpu_cost[1] = gpu_cost[1] + cst.gpu;
        std::get<1>(choices.back()) = 0;
      }
    } else {

      // we don't have a CPU kernel only a GPU kernel
      std::get<1>(choices.back()) = -1;
      cpu_cost[1] = std::numeric_limits<float>::infinity();
      gpu_cost[1] = std::numeric_limits<float>::infinity();
    }

    if (cst.is_cpu()) {
      if ((on_cpu_total + cst.cpu) < (on_gpu_total + cst.cpu)) {
        cpu_cost[0] += cst.cpu;
        std::get<0>(choices.back()) = 1;
      } else {
        cpu_cost[0] = cpu_cost[1] + cst.cpu;
        gpu_transfer[0] = gpu_transfer[1];
        gpu_cost[0] = gpu_cost[1];
        std::get<1>(choices.back()) = 0;
      }
    } else {

      // we don't have a GPU kernel only a CPU kernel
      cpu_cost[0] = std::numeric_limits<float>::infinity();
      gpu_cost[0] = std::numeric_limits<float>::infinity();
      std::get<1>(choices.back()) = -1;
    }
  }

  auto on_cpu_total = cpu_cost[0] + gpu_cost[0] + gpu_transfer[0];
  auto on_gpu_total = cpu_cost[1] + gpu_cost[1] + gpu_transfer[1];

  if (on_cpu_total < on_gpu_total) {
    backtrace(choices, 0, gpu_assigment);
    return {transfer_cost, cpu_cost[0], gpu_cost[0], gpu_transfer[0]};
  } else {
    backtrace(choices, 1, gpu_assigment);
    return {transfer_cost, cpu_cost[1], gpu_cost[1], gpu_transfer[1]};
  }
}

std::tuple<float, float, float, float>
two_layer_compiler::try_assign(
  node_id_t node, const std::list<uint32_t> &cmd,
  const std::vector<abstract_command_t> &commands,
  std::vector<std::unordered_set<tid_t>> &tensor_locations,
  std::vector<std::tuple<tid_t, node_id_t>> &history)
{
  float transfer_cost = 0.0f;
  // the tensors present on the node
  auto &present_tensors = tensor_locations[node];

  // go through all the required inputs and sum the costs
  for (auto in : commands[cmd.front()].input_tids) {
    auto cst = cost_model->get_transfer_cost(in);
    if (present_tensors.find(in) == present_tensors.end()) {
      transfer_cost += cst.network_transfer_cost;
    }
  }

  // check if the tensor is on the node
  float cpu_cost[2] = {
      0.0f, std::numeric_limits<float>::infinity()}; // 0 off the GPU
  float gpu_cost[2] = {
      0.0f, std::numeric_limits<float>::infinity()}; // 1 on the GPU
  float gpu_transfer[2] = {
      0.0f, std::numeric_limits<float>::infinity()}; // 1 on the GPU

  for (auto &c : cmd) {

    // calculate to run the kernels
    auto cst = cost_model->get_execution_cost(c);

    // calculate the cost to transfer the input to the GPU
    float gpu_transfer_cost = 0.0f;
    for (auto in : commands[c].input_tids) {
      auto tcst = cost_model->get_transfer_cost(in);
      if (present_tensors.find(in) == present_tensors.end()) {
        gpu_transfer_cost += tcst.gpu_transfer_cost;
      }
    }

    // mark the location
    for (auto out : commands[c].output_tids) {
      tensor_locations[node].insert(out);
      history.push_back({out, node});
    }

    // the total cost
    auto on_cpu_total = cpu_cost[0] + gpu_cost[0] + gpu_transfer[0];
    auto on_gpu_total = cpu_cost[1] + gpu_cost[1] + gpu_transfer[1];

    if (cst.is_gpu()) {
      if ((on_gpu_total + cst.gpu) <
          (on_cpu_total + gpu_transfer_cost + cst.gpu)) {
        gpu_cost[1] += cst.gpu;
      } else {
        cpu_cost[1] = cpu_cost[0];
        gpu_transfer[1] = gpu_transfer[0] + gpu_transfer_cost;
        gpu_cost[1] = gpu_cost[1] + cst.gpu;
      }
    } else {

      // we don't have a CPU kernel only a GPU kernel
      cpu_cost[1] = std::numeric_limits<float>::infinity();
      gpu_cost[1] = std::numeric_limits<float>::infinity();
    }

    if (cst.is_cpu()) {
      if ((on_cpu_total + cst.cpu) < (on_gpu_total + cst.cpu)) {
        cpu_cost[0] += cst.cpu;
      } else {
        cpu_cost[0] = cpu_cost[1] + cst.cpu;
        gpu_transfer[0] = gpu_transfer[1];
        gpu_cost[0] = gpu_cost[1];
      }
    } else {

      // we don't have a GPU kernel only a CPU kernel
      cpu_cost[0] = std::numeric_limits<float>::infinity();
      gpu_cost[0] = std::numeric_limits<float>::infinity();
    }
  }

  auto on_cpu_total = cpu_cost[0] + gpu_cost[0] + gpu_transfer[0];
  auto on_gpu_total = cpu_cost[1] + gpu_cost[1] + gpu_transfer[1];

  if (on_cpu_total < on_gpu_total) {
    return {transfer_cost, cpu_cost[0], gpu_cost[0], gpu_transfer[0]};
  } else {
    return {transfer_cost, cpu_cost[1], gpu_cost[1], gpu_transfer[1]};
  }
}

void two_layer_compiler::_update_present_tids(
  std::unordered_set<tid_t> &present_tids,
  const std::vector<std::list<uint32_t>> &first_layer,
  const std::vector<abstract_command_t> &commands)
{
  for (auto &cmd_lst : first_layer) {
    for (auto &cmd : cmd_lst) {
      for (auto tid : commands[cmd].output_tids) {
        present_tids.insert(tid);
      }
    }
  }
}

std::vector<std::list<uint32_t>>
two_layer_compiler::_get_layer(
  const std::vector<abstract_command_t> &commands,
  const std::unordered_set<tid_t> present_tids)
{
  // get the first layer
  std::vector<std::list<uint32_t>> first_layer;
  for (auto tid : present_tids) {

    // try to find all the commands that use this tid
    auto it = _tensor_consumers.find(tid);
    if (it == _tensor_consumers.end()) {
      continue;
    }
    auto &cmds_waiting = it->second;

    // go through all the commands waiting for this input
    auto N = cmds_waiting.size();
    for (int32_t idx = 0; idx < N; ++idx) {

      // mark the input for this command as available
      auto cmd_id = cmds_waiting[idx];
      _inputs_left[cmd_id]--;
      if (_inputs_left[cmd_id] == 0) {
        N--;
        std::swap(cmds_waiting[idx], cmds_waiting[N]);
        cmds_waiting.resize(N);

        // this command
        first_layer.push_back({cmd_id});
        idx--;
      }
    }
  }

  // figure out if we can add some more for now the idea
  // is that append ops that are unary as they don't require us to move data
  // around
  for (uint32_t idx = 0; idx < first_layer.size(); ++idx) {
    add_all_appendable(first_layer[idx], commands);
  }

  return std::move(first_layer);
}

void two_layer_compiler::add_all_appendable(
  std::list<uint32_t> &op_list,
  const std::vector<abstract_command_t> &commands)
{
  uint32_t producer = *op_list.begin();
  while (true) {

    // make sure the produced output has only one consumer and get it
    auto &p = commands[producer];
    if (p.type == abstract_command_type_t::DELETE) {
      break;
    }

    auto it = _tensor_consumers.find(p.output_tids.front());
    if (it == _tensor_consumers.end() || it->second.size() != 1) {
      break;
    }
    auto consumer = _tensor_consumers[p.output_tids.front()][0];

    // check we can actually append the op
    if (!is_apendable(producer, consumer, commands)) {
      break;
    }

    // store the consumer
    _inputs_left[consumer] = 0;
    op_list.push_back(consumer);
    producer = consumer;
  }
}

bool two_layer_compiler::is_apendable(
  uint32_t &producer, uint32_t &consumer,
  const std::vector<abstract_command_t> &commands)
{
  // get the produce and consumer references
  auto &p = commands[producer];
  auto &c = commands[consumer];

  // make sure that the producer produces exactly one output
  // exactly one input and no other command is consuming this input
  if (p.output_tids.size() != 1) {
    return false;
  }
  if (c.input_tids.size() != 1) {
    return false;
  }

  return true;
}

} // bbts

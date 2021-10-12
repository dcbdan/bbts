#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
// #include "abstract_command.h"
// #include "cost_model.h"
#include "../commands/command.h"
#include "../tensor/tensor.h"
#include "../tensor/tensor_factory.h"
#include "../ud_functions/ud_function.h"
#include "../ud_functions/udf_manager.h"


namespace bbts {

enum class abstract_command_type_t : int {
  APPLY,
  REDUCE,
  DELETE
};

using abstract_ud_spec_id_t = int32_t;

// type:APPLY  fun:fun_id in:t1,t2,t2.. out:t3,t4.. 
// type:REDUCE fun:fun_id in:t1,t2,t2.. out:t3,t4.. 
// type:APPLY  fun:fun_id in:t1,t2,t2.. out:t3,t4.. 

struct abstract_ud_spec_t {

  // the id
  abstract_ud_spec_id_t id;

  // the name of the ud fuction
  std::string ud_name;

  // the input types of the ud function
  std::vector<std::string> input_types;
 
  // the output types
  std::vector<std::string> output_types; 

  // write the function to the file
  void write_to_file(std::ofstream &file) {
    
    // write the stuff
    file << id << " " << ud_name << " " << input_types.size() << " ";
    for(auto &s : input_types) {
      file << s << " ";
    }
    file << output_types.size() << " ";
    for(auto &s : output_types) {
      file << s << " ";
    }
    file << '\n';
  }

  void read_from_file(std::ifstream &file) {

    // write the stuff
    std::size_t num_inputs;
    file >> id >> ud_name >> num_inputs;

    // read all the input types
    input_types.resize(num_inputs);
    for(auto &s : input_types) {
      file >> s;
    }

    // read the number of outputs
    std::size_t num_outputs;
    file >> num_outputs;

    // read all the outputs
    output_types.resize(num_outputs);
    for(auto &s : output_types) {
      file >> s;
    }
  }

};

class cost_model_t {
private:
  // the matched function
  struct function_match_t {

    // the cpu implementation
    ud_impl_t* cpu;

    // the gpu implementation
    ud_impl_t* gpu;
  };

public:

  // the choice we made
  struct ud_choice_t {

    // the cost
    float cost;

    // is this using the gpu
    bool is_gpu;

    // the pointer
    ud_impl_t* ud;
  };

  cost_model_t(std::unordered_map<tid_t, tensor_meta_t> meta,
               const std::vector<abstract_ud_spec_t> &funs,
               tensor_factory_ptr_t tf, 
               udf_manager_ptr manager, 
               float gpu_transfer_cost_per_byte, 
               float send_cost_per_byte) : meta(std::move(meta)),
                                           tf(std::move(tf)),
                                           udm(std::move(manager)),
                                           send_cost_per_byte(send_cost_per_byte),
                                           gpu_transfer_cost_per_byte(gpu_transfer_cost_per_byte) {
    
    
    for(const auto &f : funs) {

      // get the matcher
      auto matcher = udm->get_matcher_for(f.ud_name);
      if(matcher == nullptr) {
        throw std::runtime_error("Could not match ud " + f.ud_name);
      }

      // the inplementation
      auto impl_cpu = matcher->findMatch(f.input_types, f.output_types, false);
      auto impl_gpu = matcher->findMatch(f.input_types, f.output_types, true);

      if(impl_cpu == nullptr && impl_gpu == nullptr) {
        throw std::runtime_error("Could not find a matching ud function.");
      }

      // found the match store it
      matched_functions[f.id] = function_match_t{.cpu = impl_cpu, .gpu = impl_gpu};
    }
  }

  // figure out the kernel cost
  ud_choice_t get_ud_cost(abstract_ud_spec_id_t fn, 
                          const bbts::ud_impl_t::tensor_params_t &params,
                          const std::vector<tid_t> input_tids) {
    
    // find the implementations for this kernel
    auto impls = matched_functions[fn];

    // make the meta
    _input_meta_vec.resize(input_tids.size());
    input_meta.reinit(_input_meta_vec);
    for(size_t idx = 0; idx < input_tids.size(); ++idx) {

      
      auto it = meta.find(input_tids[idx]);
      if(it == meta.end()) {
        throw std::runtime_error("Failed to find the meta for tid " + std::to_string(input_tids[idx]) + ".");
      }

      input_meta.set(idx, it->second);
    }

    // do we only have one implementation
    if(impls.gpu == nullptr) {
      return ud_choice_t{.cost = (float) impls.cpu->get_complexity_hint(params, input_meta), .is_gpu = false, .ud = impls.cpu};
    }
    else if(impls.cpu == nullptr) {
      return ud_choice_t{.cost = (float) impls.gpu->get_complexity_hint(params, input_meta), .is_gpu = true, .ud = impls.gpu};
    }

    // figure out the cpu transfer cost
    float cpu_cost = (float) impls.cpu->get_complexity_hint(params, input_meta);

    // figure out the gpu compute cost
    float gpu_compute_cost = (float) impls.gpu->get_complexity_hint(params, input_meta);
    
    // figure out the gpu transfer cost
    float gpu_transfer_cost = 0;
    for(size_t idx = 0; idx < input_meta.num_args(); ++idx) {
      gpu_transfer_cost += gpu_transfer_cost_per_byte * this->tf->get_tensor_size(input_meta.get_by_idx(idx));
    }

    // check if we pick the cpu
    if(cpu_cost < std::max(gpu_transfer_cost, gpu_compute_cost)) {
      return ud_choice_t{.cost = cpu_cost, .is_gpu = false, .ud = impls.cpu};
    }

    // we pick the gpu
    return ud_choice_t{.cost = std::max(gpu_transfer_cost, gpu_compute_cost), .is_gpu = true, .ud = impls.gpu}; 
  } 


  // here we figure out only the avarage reduce kernel operation as it is farily hard
  // to figure out the optinal aggregation order for non-uniform reduce kernel inputs
  ud_choice_t get_reduce_cost(abstract_ud_spec_id_t fn, 
                              const bbts::ud_impl_t::tensor_params_t &params,
                              const std::vector<tid_t> input_tids,
                              const std::vector<tid_t> output_tids) {
    
    // find the implementations for this kernel
    auto impls = matched_functions[fn];

    // we only have two inputs to a single reduce op
    _input_meta_vec.resize(2);
    input_meta.reinit(_input_meta_vec);

    _output_meta_vec.resize(1);
    out_meta.reinit(_output_meta_vec);

    tensor_meta_t out_gpu = meta.find(input_tids[0])->second;
    tensor_meta_t out_cpu = out_gpu;

    float average_gpu_cost = 0;
    float average_cpu_cost = 0;

    for(size_t idx = 1; idx < input_tids.size(); ++idx) {

      // set the rhs
      auto it = meta.find(input_tids[idx]);
      if(it == meta.end()) {
        throw std::runtime_error("Failed to find the lhs meta.");
      }
      input_meta.set(0, it->second);

      // check if there is a gpu implementation
      if(impls.gpu != nullptr) {

        // set the lhs
        it = meta.find(input_tids[0]);
        if(it == meta.end()) {
          throw std::runtime_error("Failed to find the lhs meta.");
        }
        input_meta.set(0, it->second);

        // increase the gpu cost
        float cur_gpu_cost = (float) impls.cpu->get_complexity_hint(params, input_meta);

        // we need to get this out
        out_meta.set(0, out_gpu);
        impls.gpu->get_out_meta(params, input_meta, out_meta);

        // calculate the transfer cost
        float gpu_transfer_cost = 0;
        gpu_transfer_cost += gpu_transfer_cost_per_byte * this->tf->get_tensor_size(input_meta.get_by_idx(0));
        gpu_transfer_cost += gpu_transfer_cost_per_byte * this->tf->get_tensor_size(input_meta.get_by_idx(1));

        // take the greater cost
        average_gpu_cost += std::max(gpu_transfer_cost, cur_gpu_cost);
      }

      // check if there is a cpu implementation
      if(impls.cpu != nullptr) {

        // set the lhs
        it = meta.find(input_tids[0]);
        if(it == meta.end()) {
          throw std::runtime_error("Failed to find the lhs meta.");
        }
        input_meta.set(0, it->second);

        // increase the cpu cost
        average_cpu_cost += (float) impls.cpu->get_complexity_hint(params, input_meta);

        // we need to get this out
        out_meta.set(0, out_cpu);
        impls.cpu->get_out_meta(params, input_meta, out_meta);
      }
    }
    
    // we apply the ud n - 1 times
    average_cpu_cost /= input_tids.size() - 1;
    average_gpu_cost /= input_tids.size() - 1;

    // store the meta
    for(auto idx = 0; idx < output_tids.size(); ++idx) {
      meta[output_tids[idx]] = out_meta.get_by_idx(idx);
    }

    // check if we have only the cpu or it is better than the gpu
    if(impls.gpu == nullptr || average_cpu_cost < average_gpu_cost) {
      return ud_choice_t{.cost = average_cpu_cost, .is_gpu = false, .ud = impls.cpu};
    }

    // we pick the gpu
    return ud_choice_t{.cost = average_gpu_cost, .is_gpu = true, .ud = impls.gpu}; 
  }

  float get_transfer_cost(tid_t tensor) {
    return send_cost_per_byte * this->tf->get_tensor_size(meta[tensor]);
  }

  void update_meta_for_ud(abstract_ud_spec_id_t fn,
                          bool use_gpu,
                          const bbts::ud_impl_t::tensor_params_t &params,
                          const std::vector<tid_t> input_tids,
                          const std::vector<tid_t> output_tids) {

    // make the meta
    _input_meta_vec.resize(input_tids.size());
    input_meta.reinit(_input_meta_vec);
    for(size_t idx = 0; idx < input_tids.size(); ++idx) {

      auto it = meta.find(input_tids[idx]);
      if(it == meta.end()) {
        throw std::runtime_error("Failed to find the meta.");
      }

      input_meta.set(idx, it->second);
    }

    // find the implementations for this kernel
    auto impls = matched_functions[fn];

    // are we using the gpu
    if(!use_gpu) {

      // resize the out meta
      if(output_tids.size() != impls.cpu->outputTypes.size()) {
        throw std::runtime_error("");
      }

      _output_meta_vec.resize(impls.cpu->outputTypes.size());
      out_meta.reinit(_output_meta_vec);
      impls.cpu->get_out_meta(params, input_meta, out_meta);
    }
    else {

      // resize the out meta
      if(output_tids.size() != impls.gpu->outputTypes.size()) {
        throw std::runtime_error("");
      }

      _output_meta_vec.resize(impls.gpu->outputTypes.size());
      out_meta.reinit(_output_meta_vec);
      impls.gpu->get_out_meta(params, input_meta, out_meta);
    }

    // store the meta
    for(auto idx = 0; idx < output_tids.size(); ++idx) {
      meta[output_tids[idx]] = out_meta.get_by_idx(idx);
    }
  }

private:

  // all the matched functions
  std::unordered_map<abstract_ud_spec_id_t, function_match_t> matched_functions;

  // the meta so far in the cost model
  std::unordered_map<tid_t, tensor_meta_t> meta;

  // the tensor factory
  tensor_factory_ptr_t tf;

  // the udf manager
  udf_manager_ptr udm;

  // what is the extimated cost of moving bytes
  float send_cost_per_byte;

  // what is the estimated cost of moving bytes to the gpu
  float gpu_transfer_cost_per_byte;

  // we use this so that we don't keep allocating 
  ud_impl_t::meta_args_t input_meta;
  std::vector<tensor_meta_t> _input_meta_vec;

  // the output meta so that we don't keep allocating 
  ud_impl_t::meta_args_t out_meta;
  std::vector<tensor_meta_t> _output_meta_vec;
};

using cost_model_ptr_t = std::shared_ptr<cost_model_t>;


struct abstract_command_t {

  // the type 
  abstract_ud_spec_id_t ud_id;

  // the type of the command (APPLY or REDUCE)
  abstract_command_type_t type;

  // the input tids
  std::vector<tid_t> input_tids;

  // the output tids
  std::vector<tid_t> output_tids;

  // parameters
  std::vector<command_param_t> params;

  // write the command to file
  void write_to_file(std::ofstream &file) {
    
    // figure out the sting label
    std::string type_string;
    switch (type) {
      case abstract_command_type_t::APPLY  : type_string = "APPLY"; break;
      case abstract_command_type_t::REDUCE : type_string = "REDUCE"; break;
      case abstract_command_type_t::DELETE : type_string = "DELETE"; break;
    }

    // write the stuff
    file << ud_id << " " << type_string << " " << input_tids.size() << " ";
    for(auto &s : input_tids) {
      file << s << " ";
    }

    file << output_tids.size() << " ";
    for(auto &s : output_tids) {
      file << s << " ";
    }

    file << params.size() << " ";
    for(auto idx = 0; idx < params.size(); ++idx) {
      file << "int ";
      file << params[idx].i << " ";
    }
    file << '\n';
  }

  void read_from_file(std::ifstream &file) {

    std::size_t num_input_tids;
    std::size_t num_output_tids;
    std::size_t num_params;
    std::string type_string;

    file >> ud_id >> type_string >> num_input_tids;

    if(type_string == "APPLY") {
      type = abstract_command_type_t::APPLY;
    }
    else if(type_string == "REDUCE") {
      type = abstract_command_type_t::REDUCE;
    }
    else if(type_string == "DELETE") {
      type = abstract_command_type_t::DELETE;
    }
    else {
      throw std::runtime_error("Unknown type!");
    }
  
    if(type == abstract_command_type_t::REDUCE) {
      assert(num_input_tids != 0);
    }

    input_tids.resize(num_input_tids);
    for(auto &tid : input_tids) {
      file >> tid;
    }

    file >> num_output_tids;
    output_tids.resize(num_output_tids);
    for(auto &tid : output_tids) {
      file >> tid;
    }

    file >> num_params;
    std::string type;
    for(auto idx = 0; idx < num_params; ++idx) {

      // the parameter
      command_param_t param;

      // find the type
      file >> type;
      if(type == "int") {
        file >> param.i;
      }
      else if(type == "float") {
        file >> param.f;
      }
      else if(type == "uint") {
        file >> param.u;
      }
      else {
        throw std::runtime_error("Unknown param type!");
      }

      // store the parameter
      params.push_back(param);
    }
  }
};

class command_compiler_t {
public:

command_compiler_t(cost_model_ptr_t cost_model, size_t num_nodes) : cost_model(cost_model),
                                                                    num_nodes(num_nodes) {}
  
  struct node_cost_t {

    // the total cost of transfering all the nodes
    float transfer_cost = 0;

    // the total compute cost
    float compute_cost = 0;
  };

  std::vector<bbts::command_ptr_t> compile(const std::vector<abstract_command_t> &commands,
                                           std::vector<std::unordered_set<tid_t>> &tensor_locations) {
    
    float max_compute_cost = 0;
    float max_transfer_cost = 0;

    // the costs so far
    std::vector<node_cost_t> costs(num_nodes);

    // go compile all the commands
    command_id_t cur_cmd = 0;
    std::vector<bbts::command_ptr_t> out_commands;
    for(auto &c : commands) {

      // if the command is an APPLY then we need to make sure that all the tensors are on the same node
      if(c.type == abstract_command_type_t::APPLY) {

        // store the command
        generate_apply(cur_cmd,
                       out_commands,
                       c, 
                       costs,
                       tensor_locations,
                       max_compute_cost,
                       max_transfer_cost);
      }
      else if(c.type == abstract_command_type_t::REDUCE) {

        // store the command
        generate_reduce(cur_cmd,
                        out_commands,
                        c, 
                        costs,
                        tensor_locations,
                        max_compute_cost,
                        max_transfer_cost);
      }
      else {
        
        // generate the delete commands
        generate_deletes(cur_cmd,
                         out_commands,
                         c, 
                         costs,
                         tensor_locations,
                         max_compute_cost,
                         max_transfer_cost);

      }
    }

    // insert the postprocess deletions
    _insert_deletions(out_commands, cur_cmd, tensor_locations);

    // return the commands
    return std::move(out_commands);
  }

private:

  void generate_deletes(command_id_t &cur_cmd,
                        std::vector<bbts::command_ptr_t> &out_commands,
                        const abstract_command_t &c, 
                        std::vector<node_cost_t> &costs,
                        std::vector<std::unordered_set<tid_t>> &tensor_locations,
                        float &max_compute_cost,
                        float &max_transfer_cost) {
    
    for(node_id_t node = 0; node < num_nodes; ++node) {

      // check if there is something
      std::vector<command_t::tid_node_id_t> inputs;
      for(auto &in : c.input_tids) {

        // create the command and remove
        auto it = tensor_locations[node].find(in);
        if(it != tensor_locations[node].end()) {
          inputs.push_back(command_t::tid_node_id_t{.tid = in, .node = node});
          tensor_locations[node].erase(it);
        }
      }

      if(!inputs.empty()) {
        // make the apply
        auto cmd = command_t::create_delete(cur_cmd++, inputs);

        // store the command
        out_commands.push_back(std::move(cmd));
      }
    }
  }

  void generate_reduce(command_id_t &cur_cmd,
                       std::vector<bbts::command_ptr_t> &out_commands,
                       const abstract_command_t &c, 
                       std::vector<node_cost_t> &costs,
                       std::vector<std::unordered_set<tid_t>> &tensor_locations,
                       float &max_compute_cost,
                       float &max_transfer_cost) {

    // get the compute cost of running this
    command_param_list_t raw_param = {._data = c.params.data(), ._num_elements = c.params.size()};
    auto ud_info = cost_model->get_reduce_cost(c.ud_id, 
                                               bbts::ud_impl_t::tensor_params_t{._params = raw_param}, 
                                               c.input_tids,
                                               c.output_tids);

    // check if the reduce can be local
    node_id_t local_node = _get_can_be_local(c, tensor_locations);
    if(local_node != -1) {

      // init the inputs
      std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
      for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
        inputs[idx] = (command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = local_node});
      }

      // init the parameters
      std::vector<command_param_t> params(c.params.size());
      for(int32_t idx = 0; idx < c.params.size(); ++idx) {
        params[idx] = c.params[idx];
      }

      // make the apply
      auto cmd = command_t::create_reduce(cur_cmd++, 
                                          ud_info.ud->impl_id, 
                                          ud_info.is_gpu, 
                                          params, 
                                          inputs, 
                                          command_t::tid_node_id_t{.tid = c.output_tids[0], 
                                                                   .node = local_node});

      // mark that we are creating a tensor here
      tensor_locations[local_node].insert(c.output_tids[0]);
      out_commands.push_back(std::move(cmd));
      return;
    }

    // first find the best node to send the data for each input
    std::vector<node_id_t> assigned_nodes(c.input_tids.size());
    for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      auto tid = c.input_tids[idx];
      assigned_nodes[idx] = _find_node_to_fetch(tid, tensor_locations, costs);
    }

    // find the node with the cheapest compute cost as the node where we reduce to
    // it is not really a good heuristic but whatever...
    float currBestCost = std::numeric_limits<float>::max();
    node_id_t best_node = 0;

    // go through all the nodes and figure out the compute overhead
    for(node_id_t node = 0; node < num_nodes; ++node) {

      // if this node is not one of the inputs nodes skip it... TODO make this faster 
      if(std::find(assigned_nodes.begin(), assigned_nodes.end(), node) == assigned_nodes.end()) {
        continue;
      }

      // the transfer overhead
      auto compute_overhead = std::max<float>(costs[node].compute_cost + ud_info.cost - max_compute_cost, 0);
      if(compute_overhead < currBestCost) {
        currBestCost = compute_overhead;
        best_node = node;
      }
    }

    // we got some compute (do it only in this one node, technically more of them have compute by we need to load balance)
    costs[best_node].compute_cost += ud_info.cost;
 
    // init the inputs
    std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
    for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      
      // if the tensor is local no need to fetch
      auto it = tensor_locations[best_node].find(c.input_tids[idx]);
      if(it == tensor_locations[best_node].end()) {
        inputs[idx] = (command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = assigned_nodes[idx]});
        continue;
      }

      // we are fetching it
      inputs[idx] = (command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = best_node});
    }

    // init the parameters
    std::vector<command_param_t> params(c.params.size());
    for(int32_t idx = 0; idx < c.params.size(); ++idx) {
      params[idx] = c.params[idx];
    }

    // make the reduce
    auto cmd = command_t::create_reduce(cur_cmd++, 
                                        ud_info.ud->impl_id, 
                                        ud_info.is_gpu, 
                                        params, 
                                        inputs, 
                                        command_t::tid_node_id_t{.tid = c.output_tids[0], 
                                                                 .node = best_node});

    
    // mark that we are creating a tensor here
    tensor_locations[best_node].insert(c.output_tids[0]);

    // return the command
    out_commands.push_back(std::move(cmd));
  }


  void generate_apply(command_id_t &cur_cmd,
                      std::vector<bbts::command_ptr_t> &out_commands,
                      const abstract_command_t &c, 
                      std::vector<node_cost_t> &costs,
                      std::vector<std::unordered_set<tid_t>> &tensor_locations,
                      float &max_compute_cost,
                      float &max_transfer_cost) {
    
    // unique inputs
    std::vector<std::tuple<tid_t, size_t>> unique_inputs; unique_inputs.resize(10);

    // get the compute cost of running this
    command_param_list_t raw_param = {._data = c.params.data(), ._num_elements = c.params.size()};
    auto ud_info = cost_model->get_ud_cost(c.ud_id, 
                                           bbts::ud_impl_t::tensor_params_t{._params = raw_param}, 
                                           c.input_tids);
    
    // find all the unique inputs
    _find_unique_inputs(c.input_tids, unique_inputs);

    // the best node
    float currBestCost = std::numeric_limits<float>::max();
    node_id_t best_node = 0;

    // go through all the nodes and try to assign 
    for(node_id_t node = 0; node < num_nodes; ++node) {
      
      // find the transfer cost for this node
      float total_transfer_cost = 0;
      for(auto &u : unique_inputs) {
        
        // if it is not here we need to move it
        if(tensor_locations[node].find(std::get<0>(u)) == tensor_locations[node].end()) {
          total_transfer_cost += std::get<1>(u);
        }
      }

      // the transfer overhead
      auto compute_overhead = std::max<float>(costs[node].compute_cost + ud_info.cost - max_compute_cost, 0);
      auto transfer_overhead = std::max<float>(costs[node].transfer_cost + total_transfer_cost - max_transfer_cost, 0);

      // store the best
      if(currBestCost > (compute_overhead + transfer_overhead)) {
        best_node = node;
        currBestCost = compute_overhead + transfer_overhead;
      }
    }

    // we figured out the best node, now we need to make all the move ops if necessary
    for(auto &u : unique_inputs) {
        
        // if it is not here we need to move it
        if(tensor_locations[best_node].find(std::get<0>(u)) == tensor_locations[best_node].end()) {
          
          auto it = _moved_tensors.find(std::get<0>(u));
          if(it == _moved_tensors.end()) {

            // find the node to fetch from
            auto from_node = _find_node_to_fetch(std::get<0>(u), tensor_locations, costs);

            // create the move command to this node
            _moved_tensors[std::get<0>(u)] = cur_cmd;
            auto cmd = command_t::create_move(cur_cmd++, command_t::tid_node_id_t{.tid = std::get<0>(u), .node = from_node},
                                                         command_t::tid_node_id_t{.tid = std::get<0>(u), .node = best_node});

            // increase the transfer cost on this node
            costs[from_node].transfer_cost += std::get<1>(u);
            costs[best_node].transfer_cost += std::get<1>(u);

            // store the command
            out_commands.push_back(std::move(cmd));
            tensor_locations[best_node].insert(std::get<0>(u));
          }
          else  {
            
            // the command 
            auto cmd_id = it->second;

            // update the move op
            _update_move(out_commands, cmd_id, best_node);
            costs[best_node].transfer_cost += std::get<1>(u);

            // mark the location
            tensor_locations[best_node].insert(std::get<0>(u));
          }
        }
    }

    // init the inputs
    std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
    for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      inputs[idx] = command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = best_node};
    }

    // init the outputs
    std::vector<command_t::tid_node_id_t> outputs(c.output_tids.size());
    for(int32_t idx = 0; idx < c.output_tids.size(); ++idx) {

      // store the output location
      outputs[idx] = command_t::tid_node_id_t{.tid = c.output_tids[idx], .node = best_node};

      // mark that we are creating a tensor here
      tensor_locations[best_node].insert(c.output_tids[idx]);
    }

    // init the parameters
    std::vector<command_param_t> params(c.params.size());
    for(int32_t idx = 0; idx < c.params.size(); ++idx) {
      params[idx] = c.params[idx];
    }

    // make the apply
    auto cmd = command_t::create_apply(cur_cmd++, 
                                        ud_info.ud->impl_id, 
                                        ud_info.is_gpu, 
                                        params, 
                                        inputs, 
                                        outputs);
    
    // we got some compute
    costs[best_node].compute_cost += ud_info.cost;

    // update the meta for the ud function
    raw_param = {._data = c.params.data(), ._num_elements = c.params.size()};
    cost_model->update_meta_for_ud(c.ud_id, 
                                   ud_info.is_gpu, 
                                   bbts::ud_impl_t::tensor_params_t{._params = raw_param}, 
                                   c.input_tids, 
                                   c.output_tids);

    // move the command
    out_commands.push_back(std::move(cmd));
  }

  node_id_t _find_node_to_fetch(tid_t id,
                                const std::vector<std::unordered_set<tid_t>> &tensor_locations,
                                const std::vector<node_cost_t> &costs) {
    
    // init to find the best
    float currBestCost = std::numeric_limits<float>::max();
    node_id_t bestNode = -1;

    // check every node
    for(tid_t node = 0; node < num_nodes; ++node) {

      // check if the tensor is on this node
      auto it = tensor_locations[node].find(id);
      if(it != tensor_locations[node].end()) {

        // is this the best option
        auto c = costs[node].transfer_cost;
        if(c < currBestCost) {
          currBestCost = c;
          bestNode = node;
        }
      }
    }

    // return the node
    return bestNode;
  }

  node_id_t _get_can_be_local(const abstract_command_t &c,
                              std::vector<std::unordered_set<tid_t>> &tensor_locations) {
    
    // go through every node
    for(tid_t node = 0; node < num_nodes; ++node) {
      
      // check if every tensor is present on the node
      bool fine = true;
      for(auto i : c.input_tids) {
        
        if(tensor_locations[node].find(i) == tensor_locations[node].end()) {
          fine = false;
          break;
        }
      }

      // are they all on the same node
      if(fine) {
        return node;
      }
    }

    return -1;
  }

  // find all the unique input tids
  void _find_unique_inputs(const std::vector<tid_t> &input, std::vector<std::tuple<tid_t, size_t>> &output)  {
    output.clear();
    for(auto t : input) {
      if(std::find_if(output.begin(), output.end(), [&](const std::tuple<tid_t, size_t> item) { return std::get<0>(item) == t; }) == output.end()) {
        output.push_back({t, cost_model->get_transfer_cost(t)});
      }
    }
  }

  // update the move op
  void _update_move(std::vector<bbts::command_ptr_t> &out_commands, command_id_t &cmd_id, node_id_t best_node) {

    // store the previous command
    auto cmd = std::move(out_commands[cmd_id]);

    // copy the previous
    std::vector<command_t::tid_node_id_t> out; out.reserve(cmd->get_num_outputs() + 1);
    for(int32_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {
      out.push_back(cmd->get_output(idx));
    }
    out.push_back(command_t::tid_node_id_t{.tid = cmd->get_input(0).tid, .node = best_node});

    // store the command
    out_commands[cmd_id] = command_t::create_broadcast(cmd_id, cmd->get_input(0), out);
  }

  void _insert_deletions(std::vector<bbts::command_ptr_t> &out_commands, 
                         command_id_t cur_cmd,
                         std::vector<std::unordered_set<tid_t>> &tensor_locations) {
    
    // 
    std::vector<command_t::tid_node_id_t> in(1);
    for(auto &it : _moved_tensors) {
      
      // resize the inputs
      auto cmd = out_commands[it.second].get();

      // make a delete for each of them
      for(size_t idx = 0; idx < cmd->get_num_outputs(); idx++) {

        // make sure the tensor is acutally located there
        in[0] = cmd->get_output(idx);
        auto jt = tensor_locations[in[0].node].find(in[0].tid);
        if(jt != tensor_locations[in[0].node].end()) {

          // delete it
          tensor_locations[in[0].node].erase(jt);
          out_commands.push_back(command_t::create_delete(cur_cmd++, in));
        }
      }
    }
  }

  // the cost model
  cost_model_ptr_t cost_model;

  // the number of nodes
  size_t num_nodes;

  // the tensors what are moved
  std::unordered_map<tid_t, command_id_t> _moved_tensors;
};

struct compile_source_file_t {

  // specifies all the functions used
  std::vector<abstract_ud_spec_t> function_specs;

  // specifies all the commands used
  std::vector<abstract_command_t> commands;

  void write_to_file(std::ofstream &file) {

    // write number of functions
    file << function_specs.size() << "\n";

    // write the function specs
    for(auto &fn : function_specs) {
      fn.write_to_file(file);
    }

    // write the command numbers
    file << commands.size() << '\n';

    // write the commands
    for(auto &cmd : commands) {
      cmd.write_to_file(file);
    }
  }

  void read_from_file(std::ifstream &file) {

    // get the number of functions
    std::size_t num_funs;
    file >> num_funs;

    // read all the functions
    function_specs.resize(num_funs);
    for(auto &fn : function_specs) {
      fn.read_from_file(file);
    }

    // read the number of commands
    std::size_t num_cmds;
    file >> num_cmds;

    // read all the commands
    commands.resize(num_cmds);
    for(auto &cmd : commands) {
      cmd.read_from_file(file);
    }
  }

};


}

#pragma once

#include "../commands/command.h"
#include "../tensor/tensor.h"
#include "../tensor/tensor_factory.h"
#include "../ud_functions/ud_function.h"
#include "../ud_functions/udf_manager.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace bbts {

enum class abstract_command_type_t : int { APPLY, REDUCE, DELETE };

using abstract_ud_spec_id_t = int32_t;

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

    // figure out the string label
    std::string type_string;
    switch (type) {
    case abstract_command_type_t::APPLY:
      type_string = "APPLY";
      break;
    case abstract_command_type_t::REDUCE:
      type_string = "REDUCE";
      break;
    case abstract_command_type_t::DELETE:
      type_string = "DELETE";
      break;
    }

    // write the stuff
    file << ud_id << " " << type_string << " " << input_tids.size() << " ";
    for (auto &s : input_tids) {
      file << s << " ";
    }

    file << output_tids.size() << " ";
    for (auto &s : output_tids) {
      file << s << " ";
    }

    file << params.size() << " ";
    for (auto idx = 0; idx < params.size(); ++idx) {
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

    if (type_string == "APPLY") {
      type = abstract_command_type_t::APPLY;
    } else if (type_string == "REDUCE") {
      type = abstract_command_type_t::REDUCE;
    } else if (type_string == "DELETE") {
      type = abstract_command_type_t::DELETE;
    } else {
      throw std::runtime_error("Unknown type!");
    }

    if (type == abstract_command_type_t::REDUCE) {
      assert(num_input_tids != 0);
    }

    input_tids.resize(num_input_tids);
    for (auto &tid : input_tids) {
      file >> tid;
    }

    file >> num_output_tids;
    output_tids.resize(num_output_tids);
    for (auto &tid : output_tids) {
      file >> tid;
    }

    file >> num_params;
    std::string type;
    for (auto idx = 0; idx < num_params; ++idx) {

      // the parameter
      command_param_t param;

      // find the type
      file >> type;
      if (type == "int") {
        file >> param.i;
      } else if (type == "float") {
        file >> param.f;
      } else if (type == "uint") {
        file >> param.u;
      } else {
        throw std::runtime_error("Unknown param type!");
      }

      // store the parameter
      params.push_back(param);
    }
  }
};

} // namespace bbts

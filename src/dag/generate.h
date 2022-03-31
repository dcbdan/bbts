#pragma once

#include <vector>
#include <tuple>
#include <functional>

#include "dag.h"
#include "partition_info.h"
#include "../commands/command.h"

namespace bbts { namespace dag {

// Given a dag and for each node the corresponding partition_info_t,
// generate the input and run commands.
std::tuple<
  std::vector<bbts::command_ptr_t>,
  std::vector<bbts::command_ptr_t>>
    generate(
      dag_t const& dag,
      std::vector<partition_info_t> const& info,
      std::function<bbts::ud_impl_id_t(int)> get_ud,
      int num_nodes);

}}

#pragma once

#include "partition.h"

namespace bbts { namespace dag {

struct run_info_t {
  int priority;
  vector<int> blocking;
};

// For each node, get the run info by
// unleasing Gecode with Partition
vector<run_info_t> run(partition_options_t const& opt);

}}

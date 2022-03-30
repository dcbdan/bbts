#pragma once

#include "partition.h"
#include "../partition_info.h"

namespace bbts { namespace dag {

// For each node, get the run info by
// unleasing Gecode with Partition
vector<partition_info_t> run(partition_options_t const& opt);

}}

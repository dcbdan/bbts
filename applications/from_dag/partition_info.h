#pragma once

#include <vector>

namespace bbts { namespace dag {

struct partition_info_t {
  int priority;
  std::vector<int> blocking;
};

}}

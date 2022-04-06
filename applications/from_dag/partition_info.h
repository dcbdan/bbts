#pragma once

#include <vector>

namespace bbts { namespace dag {

struct partition_info_t {
  vector<int> blocking;
  int start;
  int duration;
  int worker;
};

}}

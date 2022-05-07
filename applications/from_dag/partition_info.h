#pragma once

#include <vector>

namespace bbts { namespace dag {

struct partition_info_t {
  partition_info_t(): start(-1) {}

  partition_info_t(vector<int> blocking_, int start_, int duration_, int worker_, int unit_):
    blocking(blocking_), start(start_), duration(duration_), worker(worker_), unit(unit_)
  {}

  // This is a maybe type!
  bool is_set() const { return start >= 0; }

  vector<int> blocking;
  int start;
  int duration;
  int worker;
  int unit;
};

}}

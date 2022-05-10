#pragma once

#include "dag.h"
#include "relation.h"
#include "misc.h"

#include <vector>
#include <tuple>
#include <functional>
#include <set>

namespace bbts { namespace dag {

struct placement_t {
  placement_t(): placement_t(0) {}
  placement_t(int n): computes(n), locs(n) {}

  vector<int> computes;
  vector<std::set<int>> locs;
};

vector<placement_t> greedy_solve_placement(
  vector<relation_t> const& relations,
  int num_nodes);

}}

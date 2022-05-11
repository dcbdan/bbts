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
  placement_t(int n): computes(n, -1), locs(n) {}

  placement_t(vector<int> const& computes_):
    computes(computes_), locs(computes_.size())
  {
    for(int i = 0; i != computes.size(); ++i) {
      locs[i].insert(computes[i]);
    }
  }

  int size() const {
    if(computes.size() != locs.size()) {
      throw std::runtime_error("computes and locs do not have the same size!");
    }
    return computes.size();
  }

  vector<int> computes;
  vector<std::set<int>> locs;
};

vector<vector<int>> just_computes(vector<placement_t> const& placements);

vector<placement_t> greedy_solve_placement(
  bool with_order,
  bool with_outputs,
  relations_t const& relations,
  int num_nodes);

vector<vector<int>> dumb_solve_placement(
  relations_t const& relations,
  int num_nodes);

uint64_t total_move_cost(
  relations_t const& relations,
  vector<vector<int>> const& computes);

}}

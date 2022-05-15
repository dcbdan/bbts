#pragma once

#include <unordered_map>
#include <set>
#include <memory>
#include <functional>
#include <tuple>
#include <vector>

#include "../dag.h"
#include "min_cost_tree.h"

namespace bbts { namespace dag {

using std::vector;
using std::function;
using std::tuple;

struct search_params_t {
  int num_workers;
  int max_depth;
  bool all_parts;
  bool include_outside_up_reblock;
  bool include_outside_down_reblock;
  int flops_scale_min;
  int flops_scale_max;
  int bytes_scale_min;
  int bytes_scale_max;
  int reblock_multiplier;
  int barrier_reblock_multiplier;
};

// For each possible join node,
//   Form a tree from that join node, and do a dynamic programming algorihtm
//   at that tree to determine the best partitioning.
// If depth = 1, this is a greedy algorithm.
// If Each dimension size has a lot of possible parts, the algorithm will take a long time.
// A "possible join node" is an output join or a join that only has an aggregation above it.
// It is not possible if a join has a join directly above it. Put another way: joins are
// fused.
vector<vector<int>> run_partition(
  // THIS DAG MUST BE SUPERIZED: THE INPUTS OF EVERY JOIN MUST BE REBLOCKS!
  dag_t const& dag,
  search_params_t const& params,
  std::unordered_map<int, std::vector<int>> const& possible_parts);

// Insert reblocks to make sure that the only inputs to joins are
// reblocks. This makes the graph easier to reason about since
// things like J-J constrain partitionings. In the worst case,
// something like
//    J J
//   J J J
// can happen, where you have to traverse this complicated graph
// to determine the dependencies in terms of partitioning.. Instead,
// by making sure every J only has reblock inputs, everything is easy
// to understand: every J and every I needs a partition.
// TODO(WISH): create a partiitoner where reblocks don't have to be
//             everywhere
void superize(vector<node_t>& dag);

struct solver_t {
  solver_t(
    // This dag must be so that every join's input is a reblock!
    dag_t const& dag_,
    search_params_t const& params_,
    std::unordered_map<int, vector<int>> const& possible_parts_);

  // solve the dynamic programming problem starting at nid
  void solve(nid_t nid);

  // Can you solve solve at this node?
  bool can_solve_at(nid_t nid) const { return cost_nodes[nid] != nullptr; }

  // Build up the partition for all nodes, not just the join and input nodes
  vector<vector<int>> get_partition() const;

  // TODO
  uint64_t cost() const { throw std::runtime_error("not implemented"); return 0; }

private:
  // This guy holds a tree of computations and provides the necessary info
  // for calling the tree solver
  struct coster_t {
    coster_t(nid_t, solver_t* self);

    // The options tree
    tree::tree_t< vector<vector<int>> > t_nids;

    // All the nodes in the tree
    std::set<nid_t> s_nids;

    vector<vector<int>> get_options(nid_t nid);

    // This is costing with respect to the super node, for use by the tree solver;
    // these call cost_node and cost_reblock
    uint64_t cost_super_node(nid_t        nid, vector<int> const&        partition) const;
    uint64_t cost_super_edge(nid_t parent_nid, vector<int> const& parent_partition,
                       nid_t        nid, vector<int> const&        partition) const;

    // This is coster for use by the actual node; they may use the search params in
    // the computation
    uint64_t cost_node(nid_t nid, vector<int> const& partition) const;
    uint64_t cost_reblock(nid_t nid, vector<int> const& p_up, vector<int> const& p_down) const;
  private:
    solver_t* self;

    uint64_t _cost(uint64_t total, int num_parallel) const;
  };

  // This just holds the graph of "super" nodes
  struct cost_node_t {
    cost_node_t(nid_t nid): nid(nid), downs() {}
    cost_node_t(nid_t nid, vector<int> downs): nid(nid), downs(downs) {}

    nid_t nid;
    vector<nid_t> downs;
  };
  using cost_node_ptr = std::unique_ptr<cost_node_t>;

private:
  // Store for every join and input node, the current partiiton
  vector<vector<int>> partition;

  vector<cost_node_ptr> cost_nodes;

  // All dags have the following constraints:
  // - Agg has one input, a join
  // - Reblock has one input
  // - Reblock has ont output, and it is a join
  // And note that
  // - For a Join - Join connection, they must have the same partition,
  //   i.e. belong to the same cost node.
  dag_t const& dag;
  std::unordered_map<int, vector<int>> possible_parts;

  search_params_t const params;
  vector<uint64_t> relation_bytes;
  vector<uint64_t> relation_flops;
};

// TODO(WISH): remove duplicate reblocks!

}}

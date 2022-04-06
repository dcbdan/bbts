#pragma once

#include <vector>
#include <tuple>
#include <functional>

#include "dag.h"
#include "partition_info.h"
#include "../../src/commands/command.h"

namespace bbts { namespace dag {

using std::vector;
using std::tuple;
using std::function;

// Let each node have a count. Given a list of nodes,
// always pick the node with the least count, then
// increment the count by one.
struct select_node_t {
  select_node_t(int n): counts(n, 0) {
    assert(n > 0);
  }

  // For each node, there is a score.
  // Pick from the nodes with the highest score.
  int select_score(vector<int> const& scores);

  int select_from(vector<int> const& select_from_);
private:
  vector<uint64_t> counts;
};

// A node in ::bbts::dag is a join,agg,reblock computation
// at the vertex of a compute graph.
//
// A node in ::bbts is a distinct
// compute object in a cluster.
using tid_loc_t = command_t::tid_node_id_t;
using loc_t = ::bbts::node_id_t;

struct generate_commands_t {
  generate_commands_t(
    dag_t const& dag,
    vector<partition_info_t> const& info,
    std::function<ud_impl_id_t(int)> get_ud,
    int num_nodes);

  tuple<vector<command_ptr_t>, vector<command_ptr_t>> extract();

  size_t size() const {
    return dag.size();
  }

  vector<nid_t> priority_dag_order() const {
    return dag.priority_dag_order(
      [&](nid_t nid){ return info[nid].priority; });
  }

  struct relation_t {
    relation_t(generate_commands_t* self_, nid_t nid_);

    vector<tid_loc_t> get_inputs(vector<int> const& bid);

    tid_loc_t& operator[](vector<int> const& bid);

    void print(std::ostream& os) const;

  private:
    bool _is_no_op();

    generate_commands_t* self;
    nid_t const nid;
    vector<tid_loc_t> tid_locs;
  public:
    std::vector<int> const& partition;
    bool const is_no_op;
  };

  relation_t& operator[](nid_t const& nid) {
    return relations[nid];
  }

private:
  void add_node(nid_t nid);

  int next_command_id() { return _command_id++; }
  int next_tid() { return _tid++; }

private:
  vector<relation_t> relations;

  vector<command_ptr_t> input_commands;
  vector<command_ptr_t> commands;

  dag_t const dag;
  vector<partition_info_t> const info;

  function<ud_impl_id_t(int)> get_ud;

  int num_nodes;
  select_node_t selector;

  int _command_id;
  int _tid;
};

}}

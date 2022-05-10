#pragma once

#include <vector>
#include <tuple>
#include <functional>

#include "dag.h"
#include "partition_info.h"
#include "relation.h"

#include "../../src/commands/command.h"

namespace bbts { namespace dag {

using std::vector;
using std::tuple;
using std::function;

// A node in ::bbts::dag is a join,agg,reblock computation
// at the vertex of a compute graph.
//
// A node in ::bbts is a distinct
// compute object in a cluster.
using tid_loc_t = command_t::tid_node_id_t;
using loc_t = ::bbts::node_id_t;

struct ud_info_t {
  ud_impl_id_t init;
  ud_impl_id_t expand;
  ud_impl_id_t castable_elementwise;
  ud_impl_id_t permute;
  ud_impl_id_t batch_matmul;
  ud_impl_id_t reduction;
  ud_impl_id_t unary_elementwise;
  ud_impl_id_t binary_elementwise;

  ud_impl_id_t get_join_ud(node_t::join_kernel_type j) const {
    if(j == node_t::join_kernel_type::contraction) {
      return batch_matmul;
    }
    if(j == node_t::join_kernel_type::reduction) {
      return reduction;
    }
    if(j == node_t::join_kernel_type::unary_elementwise) {
      return unary_elementwise;
    }
    if(j == node_t::join_kernel_type::binary_elementwise) {
      return binary_elementwise;
    }

    assert(false);
    return init;
  }
};

struct generate_commands_t {
  generate_commands_t(
    dag_t const& dag,
    vector<relation_t> const& relations,
    vector<vector<int>> const& compute_locs,
    ud_info_t ud_info,
    int num_nodes);

  tuple<vector<command_ptr_t>, vector<command_ptr_t>> extract();

  size_t size() const {
    return dag.size();
  }

  relation_t const& operator[](nid_t const& nid) const {
    return relations[nid];
  }

private:
  void add_node(nid_t nid);

  void add_deletes();
  void _add_deletes(nid_t nid);

  int next_command_id() { return _command_id++; }
  int next_tid()        { return _tid++;        }

  vector<tid_loc_t> get_inputs(nid_t nid, vector<int> const& bid) const;
  tid_loc_t& get_tid_loc(nid_t nid, vector<int> const& bid);

private:
  vector<vector<int>> const& compute_locs;
  vector<relation_t> const& relations;
  vector<vector<tid_loc_t>> tid_locs;

  vector<command_ptr_t> input_commands;
  vector<command_ptr_t> commands;

  std::unordered_map<tid_t, std::vector<loc_t>> moved_to_locs;
  bool was_moved_to(tid_t tid, loc_t loc);
  void assure_moved_to(
    vector<command_ptr_t>& cmds,
    tid_t tid, loc_t from, loc_t to);

  dag_t const dag;

  ud_info_t ud_info;

  int num_nodes;

  int _command_id;
  int _tid;
};

}}

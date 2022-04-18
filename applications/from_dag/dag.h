#pragma once

#include <vector>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <iostream>
#include <cassert>
#include <map>
#include <functional>

#include "../../src/utils/cache_holder.h"
#include "../../src/commands/command_utils.h"

namespace bbts { namespace dag {

using std::vector;

struct param_t {
  enum which_t {
    I, F, B
  };
 which_t  which;

  union val_t {
    int i;
    float f;
    bool b;
  };
  val_t val;

  int get_int() const {
    assert(which == which_t::I);
    return val.i;
  }
  float get_float() const {
    assert(which == which_t::F);
    return val.f;
  }
  bool get_bool() const {
    assert(which == which_t::B);
    return val.b;
  }
};

bbts::command_param_t to_bbts_param(param_t p);

std::ostream& operator<<(std::ostream& os, bbts::dag::param_t p);
std::istream& operator>>(std::istream& is, bbts::dag::param_t& p);

using nid_t  = int;
using dim_t  = int;
using rank_t = int;

// Dag assumptions:
// - Every aggs down node is a join
// - Every reblocks up node is a join
// This means you may have
// - multiple join nodes in a row
// You may not have
// - multiple aggs in a row
// - multiple reblocks in a row

struct node_t {
  enum node_type {
    input,
    reblock,
    join,
    agg
  };

  enum join_kernel_type {
    contraction,
    reduction,
    unary_elementwise,
    binary_elementwise
  };

  node_type type;

  nid_t id; // this node itself

  // The dimensions for agg and reblock nodes can be deduced
  // from the join nodes. Nevertheless, store them anyway.
  vector<dim_t> dims;

  vector<nid_t> downs;
  vector<nid_t> ups;

  vector<param_t> params;

  // A param_t can be converted intoa bbts::command_param_t.
  // Do that for each item in params and return the result.
  vector<bbts::command_param_t> get_bbts_params() const;

  // only valid if type == join
  vector<std::vector<rank_t>> ordering;
  vector<rank_t> aggs;
  join_kernel_type join_kernel;

  // Does this node "own" a partitioning
  bool is_part_owner() const {
    return type == node_type::input || type == node_type::join;
  }

private:
  friend std::ostream& operator<<(std::ostream& os, node_t const& self) {
    return self.print(os);
  }

  std::ostream& print(std::ostream& os) const;
};

struct dag_t {
  dag_t(vector<node_t> const& dag):
    dag(dag),
    _inputs(std::bind(&dag_t::_set_inputs, this)),
    _depth_dag_order(std::bind(&dag_t::_set_depth_dag_order, this)),
    _breadth_dag_order(std::bind(&dag_t::_set_breadth_dag_order, this))
  {}

  vector<node_t> const dag;

  vector<node_t> const& get_dag() const { return dag; }

  std::size_t size() const { return dag.size(); }

  node_t const& operator[](nid_t nid) const { return dag[nid]; }

  vector<nid_t> const& inputs() const {
    return _inputs();
  }

  // Given a priority value for each dag, return nodes with the smallest value
  // first, subject to being in a dag order.
  vector<nid_t> priority_dag_order(
    std::function<int(nid_t)> get_priority) const;

  vector<nid_t> const& depth_dag_order() const {
    return _depth_dag_order();
  }
  vector<nid_t> const& breadth_dag_order() const {
    return _breadth_dag_order();
  }

  // get just joins and inputs
  vector<nid_t> get_part_owners() const;

  // each node has an associated partition owner.
  // Get that nid.
  nid_t get_part_owner(nid_t nid) const;

  // filter out aggs from xs (if there are even any aggs)
  vector<int> get_out(vector<int> const& xs, nid_t nid) const;

  // filter out the non aggs (keeps) from xs
  vector<int> get_agg(vector<int> const& xs, nid_t nid) const;

  // given the join-incident dims, get the output of the reblock
  vector<int> get_out_for_input(vector<int> const& up_inc, nid_t up_id, nid_t down_id) const;

  vector<int> get_out_from_owner_incident(vector<int> const& inc, nid_t nid) const;

  // given the incdident dims for the comptue node of nid, get the corresponding
  // incident dims of the give nid
  vector<int> get_node_incident(vector<int> const& inc, nid_t nid) const;

  static vector<int> combine_out_agg(
    vector<int> const& which_aggs,
    vector<int> const& out,
    vector<int> const& agg);

private:
  // The first time these variables are called, the
  // item is set with the corresponding _set_x function.
  // Everytime after, it isn't computed again.
  cache_holder_t<vector<nid_t>> _breadth_dag_order;
  cache_holder_t<vector<nid_t>> _depth_dag_order;
  cache_holder_t<vector<nid_t>> _inputs;

  vector<nid_t> _set_breadth_dag_order();
  vector<nid_t> _set_depth_dag_order();
  vector<nid_t> _set_inputs();

  // some helper functions
  void _depth_dag_order_add_to_ret(
    vector<nid_t>& counts,
    vector<nid_t>& ret,
    nid_t id) const;
};

template <typename T>
void print_list(std::ostream& os, vector<T> xs) {
  if(xs.size() > 0) {
    os << xs[0];
  }
  for(int i = 1; i < xs.size(); ++i) {
    os << "," << xs[i];
  }
}

}}

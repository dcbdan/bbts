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

struct node_t {
  enum node_type {
    input,
    reblock,
    join,
    agg
  };

  node_type type;

  nid_t id; // this node itself

  // The dimensions for agg and reblock nodes can be deduced
  // from the join nodes. Nevertheless, store them anyway.
  vector<dim_t> dims;

  vector<nid_t> downs;
  vector<nid_t> ups;

  int kernel;
  vector<bbts::command_param_t> get_bbts_params() const;
  vector<param_t> params;

  // only valid if type == join
  vector<std::vector<rank_t>> ordering;
  vector<rank_t> aggs;

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
    _breadth_dag_order(std::bind(&dag_t::_set_breadth_dag_order, this)),
    _super_depth_dag_order(std::bind(&dag_t::_set_super_depth_dag_order, this)),
    _super_breadth_dag_order(std::bind(&dag_t::_set_super_breadth_dag_order, this))
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

  // A super node is either an input node or {[reblocks],join,agg} nodes
  vector<nid_t> const& super_depth_dag_order() const {
    return _super_depth_dag_order();
  }
  vector<nid_t> const& super_breadth_dag_order() const {
    return _super_breadth_dag_order();
  }
  vector<nid_t> super(nid_t nid) const;

  // get just joins and inputs
  vector<nid_t> get_compute_nids() const;
  vector<nid_t> get_compute_ups(nid_t) const;

  // each node has an associated compute node.
  // reblocks and aggs are attached to a join,
  // inputs are inputs.
  nid_t get_compute_nid(nid_t nid) const;

  // filter out aggs from xs (if there are even any aggs)
  vector<int> get_out(vector<int> const& xs, nid_t nid) const;

  // filter out the non aggs (keeps) from xs
  vector<int> get_agg(vector<int> const& xs, nid_t nid) const;

  // given the join-incident dims, get the output of the reblock
  vector<int> get_reblock_out(vector<int> const& join_inc, nid_t reblock_id) const;

  vector<int> get_out_from_compute(vector<int> const& inc, nid_t nid) const;

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
  cache_holder_t<vector<nid_t>> _super_breadth_dag_order;
  cache_holder_t<vector<nid_t>> _super_depth_dag_order;
  cache_holder_t<vector<nid_t>> _inputs;

  vector<nid_t> _set_breadth_dag_order();
  vector<nid_t> _set_depth_dag_order();
  vector<nid_t> _set_super_breadth_dag_order();
  vector<nid_t> _set_super_depth_dag_order();
  vector<nid_t> _set_inputs();

  // some helper functions
  void _depth_dag_order_add_to_ret(
    vector<nid_t>& counts,
    vector<nid_t>& ret,
    nid_t id) const;
  void _super_depth_dag_order_add_to_ret(
    std::map<nid_t, int>& counts,
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

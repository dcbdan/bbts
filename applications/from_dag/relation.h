#pragma once

#include <vector>
#include <tuple>
#include <functional>

#include "dag.h"
#include "partition_info.h"
#include "misc.h"

#include "../../src/utils/cache_holder.h"

namespace bbts { namespace dag {

using std::vector;
using std::tuple;
using std::function;

struct cache_t {
  cache_t()     : cache_t(0)       {}
  cache_t(int n): inns(n), outs(n) {}

  // For each nid,
  //   for each block,
  //     these are the inputs/outputs
  vector<vector<vector<tuple<nid_t, int>>>> inns;
  vector<vector<vector<tuple<nid_t, int>>>> outs;

  vector<tuple<nid_t, int>> const get_outputs(nid_t nid, int idx) const {
    return outs[nid][idx];
  }

  vector<tuple<nid_t, int>> const get_inputs(nid_t nid, int idx) const {
    return inns[nid][idx];
  }
};

struct relation_t {
  relation_t(
    nid_t nid_,
    vector<int> const& partition_,
    dag_t const& dag_,
    cache_t const& cache_,
    function<relation_t const& (nid_t)> rels_);

  // This gets the (nid, idx) pairs refering to the actual input blocks.
  vector<tuple<nid_t, int>> const&
  get_inputs(vector<int> const& bid) const {
    return cache.inns[nid][bid_to_idx(bid)];
  }

  vector<tuple<nid_t, int>> const&
  get_inputs(int idx) const {
    return cache.inns[nid][idx];
  }

  vector<tuple<nid_t, int>> const&
  get_outputs(vector<int> const& bid) const {
    return cache.outs[nid][bid_to_idx(bid)];
  }

  vector<tuple<nid_t, int>> const&
  get_outputs(int idx) const {
    return cache.outs[nid][idx];
  }

  void write_cache(cache_t& cache) const;

  // Given the bid of this guy, reach into the actual none-no-op
  // relation
  tuple<nid_t, int> operator[](vector<int> const& bid) const;

  int bid_to_idx(vector<int> const& bid) const;

  bool is_no_op() const { return _is_no_op(); }

  size_t get_num_blocks() const;

  uint64_t tensor_size() const;

  // This nid
  nid_t const nid;
  // The underlying dag
  dag_t const& dag;
  // The cache holding the connection info
  cache_t const& cache;
  // This partition
  std::vector<int> const& partition;
  // Access to all other relations
  std::function<relation_t const& ( nid_t )> rels;

private:
  cache_holder_t<bool> _is_no_op;
  bool set_is_no_op();

  // This gets the inputs, but (nid, bid) may refer to
  // a block in a no-op relation, which doesn't actually have the
  // corresponding item.
  vector<tuple<nid_t, vector<int>>>
  _get_bid_inputs(vector<int> const& bid) const;

  // This gets the (nid, idx) pairs refering to the actual input blocks.
  vector<tuple<nid_t, int>>
  _get_inputs(vector<int> const& bid) const;

};

struct relations_t {
  relations_t(
    dag_t const& dag_,
    vector<vector<int>> const& partition);

  relation_t const& operator[](nid_t nid) const {
    return relations[nid];
  }

  int size() const { return relations.size(); }

private:
  dag_t const& dag;

  cache_t cache;
  vector<relation_t> relations;
};


}}

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

struct relation_t {
  relation_t(
    nid_t nid_,
    vector<int> const& partition_,
    dag_t const& dag_,
    function<relation_t const& (nid_t)> rels_);

  // This gets the inputs, but (nid, bid) may refer to
  // a block in a no-op relation, which doesn't actually have the
  // corresponding item.
  vector<tuple<nid_t, vector<int>>>
  get_bid_inputs(vector<int> const& bid) const;

  // This gets the (nid, idx) pairs refering to the actual input blocks.
  vector<tuple<nid_t, int>>
  get_inputs(vector<int> const& bid) const;

  // Given the bid of this guy, reach into the actual none-no-op
  // relation
  tuple<nid_t, int> operator[](vector<int> const& bid) const;

  int bid_to_idx(vector<int> const& bid) const;

  bool is_no_op() const { return _is_no_op(); }

  size_t get_num_blocks() const;

  // This nid
  nid_t const nid;
  // The underlying dag
  dag_t const& dag;
  // This partition
  std::vector<int> const& partition;
  // Access to all other relations
  std::function<relation_t const& ( nid_t )> rels;

private:
  cache_holder_t<bool> _is_no_op;
  bool set_is_no_op();
};

}}

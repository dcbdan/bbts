#include "relation.h"

#include "../../src/utils/expand_indexer.h"

namespace bbts { namespace dag {

#define DCB01(x) // std::cout << "relation.cc " << __LINE__ << " " << x << std::endl

using utils::expand::expand_indexer_t;
using utils::expand::column_major_expand_t;

relations_t::relations_t(
  dag_t const& dag_,
  vector<vector<int>> const& partition):
    dag(dag_), cache(dag_.size())
{
  std::function<relation_t const& (nid_t)> get_rel = std::bind(
    &relations_t::operator[], this, std::placeholders::_1);

  relations.reserve(dag.size());
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    relations.emplace_back(
      nid, partition[nid], dag, cache, get_rel);
  }

  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    if(!relations[nid].is_no_op()) {
      auto num_blk = relations[nid].get_num_blocks();
      cache.inns[nid].resize(num_blk);
      cache.outs[nid].resize(num_blk);
    }
  }

  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    if(!relations[nid].is_no_op()) {
      relations[nid].write_cache(cache);
    }
  }
}

relation_t::relation_t(
  nid_t nid_,
  vector<int> const& partition_,
  dag_t const& dag_,
  cache_t const& cache_,
  function<relation_t const& (nid_t)> rels_):
    nid(nid_),
    partition(partition_),
    dag(dag_),
    cache(cache_),
    rels(rels_),
    _is_no_op(std::bind(&relation_t::set_is_no_op, this))
{}

vector<tuple<nid_t, int>> relation_t::_get_inputs(vector<int> const& bid) const
{
  vector<tuple<nid_t, int>> ret;
  auto items = _get_bid_inputs(bid);
  ret.reserve(items.size());

  for(auto const& [input_nid, input_bid]: items) {
    ret.push_back(rels(input_nid)[input_bid]);
  }
  return ret;
}

void relation_t::write_cache(cache_t& cache) const {
  indexer_t indexer(partition);
  do {
    auto const& bid = indexer.idx;
    int idx = bid_to_idx(bid);

    cache.inns[nid][idx] = _get_inputs(bid);

    for(auto const& [input_nid, input_idx]: cache.inns[nid][idx]) {
      cache.outs[input_nid][input_idx].emplace_back(nid, idx);
    }
  } while (indexer.increment());
}


vector<tuple<nid_t, vector<int>>>
relation_t::_get_bid_inputs(vector<int> const& bid) const
{
  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::agg) {
    vector<tuple<nid_t, vector<int>>> ret;

    // basically, cartesian product over the agg dimensions.
    nid_t join_nid = node.downs[0];
    node_t const& join_node = dag[join_nid];
    auto const& aggs = join_node.aggs;
    auto const& join_blocking = rels(join_nid).partition;

    vector<int> agg_szs;
    agg_szs.reserve(aggs.size());
    for(int which: aggs) {
      agg_szs.push_back(join_blocking[which]);
    }

    indexer_t indexer(agg_szs);

    ret.reserve(product(agg_szs));
    do {
      auto inc_bid = dag_t::combine_out_agg(aggs, bid, indexer.idx);
      ret.emplace_back(join_nid, inc_bid);
    } while(indexer.increment());

    return ret;
  }

  if(node.type == node_t::node_type::reblock) {
    nid_t input_nid = node.downs[0];

    if(is_no_op()) {
      return {{input_nid, bid}};
    } else {
      vector<tuple<nid_t, vector<int>>> ret;
      // the expander can figure out which inputs touch
      // this output.
      expand_indexer_t e_indexer(
        // this is the input partitioning
        rels(input_nid).partition,
        // this is the output partitioning
        partition);

      // get all those inputs
      auto inputs = expand_indexer_t::cartesian(e_indexer.get_inputs(bid));

      ret.reserve(inputs.size());
      for(auto const& input_bid: inputs) {
        ret.emplace_back(input_nid, input_bid);
      }
      return ret;
    }
  }

  if(node.type == node_t::node_type::mergesplit) {
    // This is the same as the reblock case, except with the added complication
    // of working around merge and split.

    nid_t input_nid = node.downs[0];

    if(is_no_op()) {
      if(node.is_merge) {
        // ij->k; is no op means 1Kij -> Kk ... this means the input
        // just needs a 0 added to the front
        return {{input_nid, expand0(bid)}};
      } else {
        // k->ij; is no op means Kk -> 1Kij ... this means that the input
        // doesn't need the zero front
        return {{input_nid, squeeze(bid)}};
      }
    }

    vector<int> inn_partition;
    vector<int> out_partition;
    vector<int> bid_fixed;
    if(node.is_merge) {
      // IJij -> Kk (= 1K1k)
      inn_partition = rels(input_nid).partition;
      out_partition = expand1(partition);
      bid_fixed     = expand0(bid);
    } else {
      // Kk (= 1K1k) -> IJij
      inn_partition = expand1(rels(input_nid).partition);
      out_partition = partition;
      bid_fixed     = bid;
    }

    vector<tuple<nid_t, vector<int>>> ret;
    // the expander can figure out which inputs touch
    // this output.
    expand_indexer_t e_indexer(inn_partition, out_partition);

    // get all those inputs
    auto inputs = expand_indexer_t::cartesian(e_indexer.get_inputs(bid_fixed));

    ret.reserve(inputs.size());
    if(node.is_merge) {
      for(auto const& input_bid: inputs) {
        ret.emplace_back(input_nid, input_bid);
      }
    } else {
      // In the split case, squeeze out the dummy dimension
      for(auto const& input_bid: inputs) {
        ret.emplace_back(input_nid, squeeze(input_bid));
      }
    }
    return ret;
  }

  if(node.type == node_t::node_type::join) {
    vector<tuple<nid_t, vector<int>>> ret;
    ret.reserve(node.downs.size());

    for(auto const& input_nid: node.downs) {
      vector<int> input_bid = dag.get_out_for_input(bid, nid, input_nid);
      ret.emplace_back(input_nid, input_bid);
    }

    return ret;
  }

  if(node.type == node_t::node_type::input) {
    return {};
  }

  throw std::runtime_error("should not reach");
  return {};
}

tuple<nid_t, int>
relation_t::operator[](vector<int> const& bid) const
{
  if(bid.size() != partition.size()) {
    throw std::runtime_error("bid size is incorrect");
  }

  if(is_no_op()) {
    node_t const& node = dag[nid];

    if(node.type == node_t::node_type::reblock) {
      // if this is a no op, the input relation has the same shape,
      // so use the same bid
      nid_t input_nid = node.downs[0];
      return rels(input_nid)[bid];
    }

    if(node.type == node_t::node_type::mergesplit) {
      // if this is a no op, the input relation has the same shape sans some ones and
      // zeros...
      nid_t input_nid = node.downs[0];
      if(node.is_merge) {
        // 1Kij -> Kk
        return rels(input_nid)[expand0(bid)];
      } else {
        // Kk -> 1Kij
        return rels(input_nid)[squeeze(bid)];
      }
    }

    if(node.type == node_t::node_type::agg) {
      // if this is a no op, then the agg dims are all 1,
      // so the inc_bid needs to be zero there.. so if j is agg,
      // ijk->ik, then bid = (1,3) -> inc_bid = (1,0,3)
      nid_t join_nid = node.downs[0];
      node_t const& join_node = dag[join_nid];

      // just in case
      if(join_node.type != node_t::node_type::join) {
        throw std::runtime_error("should not happen");
      }

      auto const& aggs = join_node.aggs;

      vector<int> inc_bid = dag_t::combine_out_agg(
        aggs,
        bid,
        vector<int>(aggs.size(), 0));

      return rels(join_nid)[inc_bid];
    }

    throw std::runtime_error("should not reach");
  }

  // otherwise, it lives here
  return {nid, bid_to_idx(bid)};
}

int relation_t::bid_to_idx(vector<int> const& bid) const
{
  int idx = 0;
  int p = 1;
  for(int r = 0; r != bid.size(); ++r) {
    idx += p * bid[r];
    p   *= partition[r];
  }
  return idx;
}

uint64_t relation_t::incident_size() const {
  vector<int> const& inc_dims = dag[nid].dims;
  uint64_t ret = 1;
  for(int i = 0; i != inc_dims.size(); ++i) {
    ret *= (inc_dims[i] / partition[i]);
  }
  return ret;
}

uint64_t relation_t::tensor_size() const {
  vector<int> const& _inc_dims = dag[nid].dims;
  vector<int> dims = dag.get_out(_inc_dims, nid);
  uint64_t ret = 1;
  for(int i = 0; i != dims.size(); ++i) {
    ret *= (dims[i] / partition[i]);
  }
  return ret;
}

size_t relation_t::get_num_blocks() const {
  if(is_no_op()) {
    return 0;
  }
  return product(partition);
}

bool relation_t::set_is_no_op() {
  node_t const& node = dag[nid];
  if(node.type == node_t::node_type::input) {
    return false;
  }
  if(node.type == node_t::node_type::join) {
    return false;
  }
  if(node.type == node_t::node_type::reblock) {
    // if these are the same, there is no reblock
    auto const& my_blocking    = partition;
    auto const& input_blocking = rels(node.downs[0]).partition;
    return my_blocking == input_blocking;
  }
  if(node.type == node_t::node_type::mergesplit) {
    // 1Jij <-> Kk <-> 1K1k
    auto const& my_blocking    = partition;
    auto const& input_blocking = rels(node.downs[0]).partition;
    if(node.is_merge) {
      // ij->k
      if(input_blocking[0] != 1) {
        return false;
      }
      return squeeze(input_blocking) == my_blocking;
    } else {
      // k->ij; split
      if(my_blocking[0] != 1) {
        return false;
      }
      return squeeze(my_blocking) == input_blocking;
    }
  }
  if(node.type == node_t::node_type::agg) {
    auto const& my_blocking    = rels(nid).partition;
    auto const& join_blocking  = rels(node.downs[0]).partition;
    // if there is nothing to agg, this is a no op
    return product(my_blocking) == product(join_blocking);
  }
  throw std::runtime_error("should not reach");
  return true;
}

vector<uint64_t> relation_t::input_tensor_sizes(vector<int> const& bid) const {
  auto const& node = dag[nid];
  if(node.type == node_t::node_type::reblock ||
     node.type == node_t::node_type::mergesplit)
  {
    vector<int> inn_partition;
    vector<int> out_partition;
    vector<int> dims;
    vector<int> bid_fixed;
    if(node.type == node_t::node_type::reblock) {
      inn_partition = rels(node.downs[0]).partition;
      out_partition = partition;
      dims = node.dims;
      bid_fixed = bid;
    } else {
      auto const& input_nid = node.downs[0];
      if(node.is_merge) {
        // IJij -> Kk (= 1K1k)
        inn_partition = rels(input_nid).partition;
        out_partition = expand1(partition);
        dims = expand1(node.dims);
        bid_fixed     = expand0(bid);
      } else {
        // Kk (= 1K1k) -> IJij
        inn_partition = expand1(rels(input_nid).partition);
        out_partition = partition;
        dims = node.dims;
        bid_fixed     = bid;
      }
    }
    vector<uint64_t> ret;
    expand_indexer_t e_indexer(inn_partition, out_partition);

    auto inputs = expand_indexer_t::cartesian(e_indexer.get_inputs(bid_fixed));
    ret.reserve(inputs.size());
    for(auto const& which_inn: inputs) {
      ret.push_back(1);
      // take the product of the expand dim sizes
      for(auto expand_dim: e_indexer.get_expand_dim(dims, which_inn, bid_fixed)) {
        ret.back() *= expand_dim.interval;
      }
    }
    return ret;
  } else {
    vector<uint64_t> ret;
    auto inputs = get_inputs(bid);
    ret.reserve(inputs.size());
    for(auto [input_nid, _0]: get_inputs(bid)) {
      ret.push_back(rels(input_nid).tensor_size());
    }
    return ret;
  }
}

}}

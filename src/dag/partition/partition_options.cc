#include "partition.h"

#include <random>
#include <functional>
#include "../misc.h"

namespace bbts { namespace dag {

using namespace Gecode;

vector<vector<int>> cartesian(
  vector<vector<int> > const& vs,
  std::function<bool(vector<int> const&)> const& should_include)
{
  // assuming all vs have a size greater than zero

  vector<int> max;
  for(auto const& v: vs) {
    max.push_back(v.size());
  }
  indexer_t indexer(max);

  vector<vector<int>> ret;
  do {
    vector<int> next_val;
    next_val.reserve(vs.size());
    for(int which_v = 0; which_v != vs.size(); ++ which_v) {
      next_val.push_back(vs[which_v][indexer.idx[which_v]]);
    }
    if(should_include(next_val)) {
      ret.push_back(next_val);
    }
  } while(indexer.increment());

  return ret;
}

vector<vector<int>> cartesian(vector<vector<int> > const& vs) {
  return cartesian(vs, [](vector<int> const&){ return true; });
}

struct random_select_t {
  random_select_t(int seed): gen(seed) {}

  template <typename T>
  T operator()(std::vector<T> const& xs) {
    std::uniform_int_distribution<> distrib(0, xs.size()-1);
    return xs[distrib(gen)];
  }

  std::mt19937 gen;
};

int partition_options_t::get_restart_scale() const {
  return _restart_scale;
}

int partition_options_t::seed() const {
  return _seed;
}

IntPropLevel partition_options_t::ipl() const {
  return _ipl;
}

int partition_options_t::get_num_workers() const {
  return _num_workers;
}

double partition_options_t::get_flops_per_time() const {
  return _flops_per_time;
}

vector<int> partition_options_t::get_all_blocks() const {
  // TODO: make this an option
  return {1,2,4,8,16,24,32,48,64,72,96,120,128,144,168,192};
}

int partition_options_t::get_min_cost() const {
  // This amount of work means that each worker can have something to do
  if(_min_cost < 0) {
    return get_num_workers();
  } else {
    return _min_cost;
  }
}

bool partition_options_t::breadth_order() const {
  return _breadth_order;
}

bool partition_options_t::cover() const {
  return _cover;
}

int partition_options_t::cover_size() const {
  return static_cast<int>(_cover_size);
}

int partition_options_t::search_compute_threads() const {
  return static_cast<int>(_search_compute_threads);
}

int partition_options_t::search_restart_scale() const {
  return static_cast<int>(_search_restart_scale);
}

double partition_options_t::search_time_per_cover() const {
  return static_cast<double>(_search_time_per_cover);
}

int partition_options_t::raw_flops(vector<int> const& dims) const {
  float flops = 1.0;
  for(auto d: dims) {
    flops *= (d*1.0);
  }
  return lround(flops / get_flops_per_time());
}

int partition_options_t::get_compute_cost(int raw_flops_cost) const {
  return get_min_cost() + raw_flops_cost;
}

int partition_options_t::get_compute_cost(vector<int> const& dims) const {
  return get_min_cost() + raw_flops(dims);
}

vector<dim_t> partition_options_t::single_partition(nid_t nid) const {
  return vector<dim_t>(dag[nid].dims.size(), 1);
}

vector<dim_t> partition_options_t::max_partition(nid_t nid) const {
  vector<dim_t> ret;
  int val = -1;
  for(auto p: all_partitions(nid)) {
    int v = product(p);
    if(v > val) {
      ret = p;
      val = v;
    }
  }
  return ret;
}

vector<vector<dim_t>> const& partition_options_t::all_partitions(nid_t nid) const {
  return _all_partitions_per_node()[nid];
}


vector<vector<vector<dim_t>>>
partition_options_t::_set_all_partitions_per_node()
{
  using  _all_partitions = vector<vector<dim_t>>;
  vector<_all_partitions> ret(dag.size());

  for(nid_t id = 0; id != dag.size(); ++id) {
    vector<vector<dim_t>> choices;
    for(auto const& dim: dag[id].dims) {
      choices.push_back(all_blocks(dim));
    }
    ret[id] = cartesian(
      choices,
      [this](vector<int> const& v)
      {
        return product(v) <= get_num_workers();
      });
    // sort it so that more workers come first
    std::sort(
      ret[id].begin(),
      ret[id].end(),
      [](vector<dim_t> const& lhs, vector<dim_t> const& rhs) {
        return product(lhs) > product(rhs);
      });
  }

  return ret;
}

vector<dim_t> const& partition_options_t::get_which_partition(nid_t nid, int which) const {
  return all_partitions(nid)[which];
}

vector<dim_t> partition_options_t::all_blocks(dim_t dim) const {
  vector<dim_t> ret;
  for(auto b: get_all_blocks()) {
    if(dim >= b && dim % b == 0 && b <= get_num_workers()) {
      ret.push_back(b);
    }
  }
  return ret;
}

vector<int> partition_options_t::time_to_completion(vector<int> const& times) const {
  vector<int> ret(dag.size());
  for(nid_t const& nid: breadth_dag_order()) {
    node_t const& node = dag[nid];

    // we're here when are inputs are all finished
    int time_to_get_here = 0;
    for(nid_t const& down: node.downs) {
      time_to_get_here = std::max(time_to_get_here, ret[down]);
    }

    int time_to_finish_here = times[nid] + time_to_get_here;
    ret[nid] = time_to_finish_here;
  }
  return ret;
}

vector<vector<dim_t>> partition_options_t::get_random_full_dag_partitioning() const {
  random_select_t random_select(seed());
  auto f = [&](nid_t nid){ return random_select(this->all_partitions(nid)); };
  return _get_partitioning(f);
}

vector<vector<dim_t>> partition_options_t::get_max_full_dag_partitioning() const {
  auto f = [&](nid_t nid){ return this->max_partition(nid); };
  return _get_partitioning(f);
}

vector<dim_t>
partition_options_t::get_kernel_inc_dims(
  vector<dim_t> const& parts,
  nid_t nid) const
{
  vector<dim_t> ret;
  vector<dim_t> const& dims = dag[nid].dims;
  for(int i = 0; i != dims.size(); ++i) {
    ret.push_back(dims[i] / parts[i]);
  }
  return ret;
}

tuple<int, int>
partition_options_t::get_est_duration_worker_pair(
  vector<int> const& inc_part,
  nid_t nid) const
{
  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::input) {
    return {0,0};
  }
  if(node.type == node_t::node_type::join) {
    return {
      get_compute_cost(get_kernel_inc_dims(inc_part, nid)), // duration
      product(inc_part)                                     // num workers
    };
  }
  if(node.type == node_t::node_type::agg) {
    auto agg_nid  = nid;
    auto join_nid = node.downs[0];
    auto agg_part = get_agg(inc_part, join_nid);
    auto out_part = get_out(inc_part, join_nid);

    auto num_aggs = product(agg_part);
    if(num_aggs == 1) {
      return {0, 0};
    } else {
      // get the ouptut kernel dims
      auto kernel_dims = get_kernel_inc_dims(out_part, agg_nid);
      // now to compute the cost, add the total number of additions
      // that has to occur, which can be done by adding to kernel_dims.
      // This is, adding 2 (1024,1024) tensors is
      //    1024*1024*1 number of adds
      kernel_dims.push_back(num_aggs-1);
      return {
        get_compute_cost(kernel_dims),   // duration
        product(out_part)                // num workers
      };
    }
  }
  if(node.type == node_t::node_type::reblock) {
    // there is no way to check if a reblock will happen without other
    // node info. So this function just returns the info as if a computation
    // is required
    auto out_part = get_reblock_out(inc_part, nid);
    return {
      get_compute_cost(get_kernel_inc_dims(out_part, nid)), // duration
      product(out_part)                                     // num workers
    };
  }

  throw std::runtime_error("get_est_duration_worker_part: invalid value for node type");
  return {0,0};
}

tuple<int, int> partition_options_t::get_duration_worker_pair(
  vector<vector<int>> const& all_parts,
  nid_t nid) const
{
  return get_duration_worker_pair(
           [&all_parts](nid_t which) { return all_parts[which]; },
           nid);
}

tuple<int, int> partition_options_t::get_duration_worker_pair(
  std::function<vector<int>(nid_t)> f,
  nid_t nid) const
{
  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::input) {
    return get_est_duration_worker_pair(f(nid), nid);
  }
  if(node.type == node_t::node_type::join) {
    return get_est_duration_worker_pair(f(nid), nid);
  }
  if(node.type == node_t::node_type::agg) {
    nid_t join_nid = node.downs[0];
    return get_est_duration_worker_pair(f(join_nid), nid);
  }
  if(node.type == node_t::node_type::reblock) {
    auto inc_part = f(node.ups[0]);
    auto curr_block = get_reblock_out(inc_part, nid);
    auto down_block = f(node.downs[0]);
    if(down_block == curr_block) {
      return {0,0};
    } else {
      return get_est_duration_worker_pair(inc_part, nid);
    }
  }
  throw std::runtime_error("get_duration_worker_part: invalid value for node type");
  return {0,0};
}

vector<vector<dim_t>> partition_options_t::_get_partitioning(
  std::function<vector<dim_t>(nid_t)> f) const
{
  vector<vector<dim_t>> ret(dag.size());
  for(node_t const& node: dag) {
    if(node.type == node_t::node_type::input) {
      // input is easy: call the input function
      nid_t nid = node.id;
      ret[nid] = f(nid);
    } else if(node.type == node_t::node_type::join) {
      // for the join incident partition, call the input function
      nid_t nid = node.id;
      ret[nid] = f(nid);
      // for the up aggs and down reblocks, deduce the partition
      vector<dim_t> const& inc_parts = ret[nid];

      // the agg has just he output part of this join
      ret[node.ups[0]] = get_out(inc_parts, nid);
      // for each of the inputs, get the value
      for(nid_t reblock_nid: node.downs) {
        ret[reblock_nid] = get_reblock_out(inc_parts, reblock_nid);
      }
    }
  }
  return ret;
}

}}

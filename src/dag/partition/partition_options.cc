#include "partition.h"

#include <random>
#include <functional>

namespace bbts { namespace dag {

template <typename T>
T product(vector<T> const& xs) {
  T ret = 1;
  for(auto x: xs) {
    ret *= x;
  }
  return ret;
}

struct indexer_t {
  indexer_t(vector<int> max):
    max(max), idx(max.size())
  {}

  bool increment(){
    bool could_increment = false;
    for(int i = 0; i != max.size(); ++i) {
      if(idx[i] + 1 == max[i]) {
        idx[i] = 0;
      } else {
        idx[i] += 1;
        could_increment = true;
        break;
      }
    }
    return could_increment;
  }

  vector<int> max;
  vector<int> idx;
};

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

int PartitionOptions::get_restart_scale() const {
  return static_cast<int>(_restart_scale.value());
}

int PartitionOptions::seed() const {
  return static_cast<int>(_seed.value());
}

IntPropLevel PartitionOptions::ipl() const {
  return _ipl.value();
}

const char* PartitionOptions::get_dag_file() const {
  return _dag_file.value();
}

const char* PartitionOptions::get_out_file() const {
  return _output_file.value();
}

const char* PartitionOptions::get_usage_file() const {
  return _usage_file.value();
}

vector<node_t> const& PartitionOptions::get_dag() const {
  return dag;
}

int PartitionOptions::get_num_workers() const {
  return static_cast<int>(_num_workers.value());
}

double PartitionOptions::get_flops_per_time() const {
  return _flops_per_time.value();
}

vector<int> PartitionOptions::get_all_blocks() const {
  // TODO: make this an option
  return {1,2,4,8,16,24,32,48,64,72,96,120,128,144,168,192};
}

int PartitionOptions::get_min_cost() const {
  // This amount of work means that each worker can have something to do
  if(_min_cost.value() < 0) {
    return get_num_workers();
  } else {
    return _min_cost.value();
  }
}

bool PartitionOptions::breadth_order() const {
  return _breadth_order.value();
}

bool PartitionOptions::cover() const {
  return _cover.value();
}

int PartitionOptions::cover_size() const {
  return static_cast<int>(_cover_size.value());
}

int PartitionOptions::search_compute_threads() const {
  return static_cast<int>(_search_compute_threads.value());
}

int PartitionOptions::search_restart_scale() const {
  return static_cast<int>(_search_restart_scale.value());
}

double PartitionOptions::search_time_per_cover() const {
  return static_cast<double>(_search_time_per_cover.value());
}

int PartitionOptions::raw_flops(vector<int> const& dims) const {
  float flops = 1.0;
  for(auto d: dims) {
    flops *= (d*1.0);
  }
  return lround(flops / get_flops_per_time());
}

int PartitionOptions::get_compute_cost(int raw_flops_cost) const {
  return get_min_cost() + raw_flops_cost;
}

int PartitionOptions::get_compute_cost(vector<int> const& dims) const {
  return get_min_cost() + raw_flops(dims);
}

void PartitionOptions::_set_dag_orders() {
  {
    _inputs = vector<nid_t>();
    for(node_t const& node: dag) {
      if(node.type == node_t::node_type::input) {
        _inputs.push_back(node.id);
      }
    }
  }

  {
    vector<nid_t> counts(dag.size());
    vector<nid_t> inputs;
    for(nid_t id = 0; id != dag.size(); ++id) {
      counts[id] = dag[id].downs.size();
      if(counts[id] == 0) {
        inputs.push_back(id);
      }
    }

    _depth_dag_order = vector<nid_t>();

    for(nid_t const& input: inputs) {
      // recursively add everything
      _depth_dag_order_add_to_ret(counts, _depth_dag_order, input);
    }
  }

  {
    vector<nid_t> counts(dag.size());
    vector<nid_t> pending;

    for(nid_t id = 0; id != dag.size(); ++id) {
      counts[id] = dag[id].downs.size();
      if(counts[id] == 0) {
        pending.push_back(id);
      }
    }

    _breadth_dag_order = vector<nid_t>();
    auto add_to_ret = [&](nid_t id) {
      _breadth_dag_order.push_back(id);
      for(nid_t const& up: dag[id].ups) {
        counts[up]--;
        if(counts[up] == 0) {
          pending.push_back(up);
        }
      }
    };

    for(int idx = 0; idx != pending.size(); ++idx) {
      add_to_ret(pending[idx]);
    }
  }

  {
    std::map<nid_t, int> counts;
    vector<nid_t> inputs;
    for(nid_t nid: get_compute_nids()) {
      // the number of compute/super children equals the number of children for compute
      // nodes.
      int num_down = dag[nid].downs.size();
      if(num_down > 0) {
        counts[nid] = num_down;
      }
      if(num_down == 0) {
        inputs.push_back(nid);
      }
    }
    _super_depth_dag_order = vector<nid_t>();
    for(nid_t const& input: inputs) {
      _super_depth_dag_order_add_to_ret(counts, _super_depth_dag_order, input);
    }
  }

  {
    std::map<nid_t, int> counts;
    vector<nid_t> pending;
    for(nid_t nid: get_compute_nids()) {
      // the number of compute/super children equals the number of children for compute
      // nodes.
      int num_down = dag[nid].downs.size();
      if(num_down > 0) {
        counts[nid] = num_down;
      }
      if(num_down == 0) {
        pending.push_back(nid);
      }
    }

    _super_breadth_dag_order = vector<nid_t>();
    auto add_to_ret = [&](nid_t nid) {
      _super_breadth_dag_order.push_back(nid);
      for(nid_t const& up: get_compute_ups(nid)) {
        counts.at(up)--; // do bounds checking just in case
        if(counts[up] == 0) {
          pending.push_back(up);
        }
      }
    };

    for(int idx = 0; idx != pending.size(); ++idx) {
      add_to_ret(pending[idx]);
    }
  }
}

vector<nid_t> PartitionOptions::get_compute_ups(nid_t nid) const {
  // make sure we have a compute nid
  nid = get_compute_nid(nid);
  node_t const& node = dag[nid];

  // get the compute ups
  vector<nid_t> ups;
  if(node.type == node_t::node_type::input) {
    for(nid_t direct_up: node.ups) {
      ups.push_back(get_compute_nid(direct_up));
    }
  } else {
    // we have a join, so reach past the agg
    for(nid_t direct_up: dag[node.ups[0]].ups) {
      ups.push_back(get_compute_nid(direct_up));
    }
  }

  return ups;
}

vector<nid_t> const& PartitionOptions::inputs() const {
  return _inputs;
}

vector<nid_t> const& PartitionOptions::depth_dag_order() const {
  return _depth_dag_order;
}

vector<nid_t> const& PartitionOptions::breadth_dag_order() const {
  return _breadth_dag_order;
}

vector<nid_t> const& PartitionOptions::super_depth_dag_order() const {
  return _super_depth_dag_order;
}

vector<nid_t> const& PartitionOptions::super_breadth_dag_order() const {
  return _super_breadth_dag_order;
}

vector<nid_t> PartitionOptions::super(nid_t nid) const {
  if(dag[nid].type == node_t::node_type::input) {
    return {nid};
  }

  nid_t join_nid = get_compute_nid(nid);
  node_t const& join_node = dag[join_nid];

  vector<nid_t> ret = join_node.downs;    // the reblocks
  ret.push_back(join_nid);                // the join
  ret.push_back(join_node.ups[0]);        // the agg

  return ret;
}

vector<nid_t> PartitionOptions::get_compute_nids() const {
  vector<nid_t> ret;
  for(node_t const& node: dag) {
    if(node.type == node_t::node_type::join ||
       node.type == node_t::node_type::input)
    {
      ret.push_back(node.id);
    }
  }
  return ret;
}

nid_t PartitionOptions::get_compute_nid(nid_t nid) const {
  node_t const& node = dag[nid];
  if(node.type == node_t::node_type::join ||
     node.type == node_t::node_type::input)
  {
    return nid;
  }
  if(node.type == node_t::node_type::reblock) {
    return node.ups[0];
  }
  if(node.type == node_t::node_type::agg) {
    return node.downs[0];
  }
  throw std::runtime_error("get_compute_nid: invalid node type");
  return -1;
}

vector<dim_t> PartitionOptions::single_partition(nid_t nid) const {
  return vector<dim_t>(dag[nid].dims.size(), 1);
}

vector<dim_t> PartitionOptions::max_partition(nid_t nid) const {
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

vector<vector<dim_t>> const& PartitionOptions::all_partitions(nid_t nid) const {
  return _all_partitions_per_node[nid];
}

void PartitionOptions::_set_all_partitions_per_node() {
  using  _all_partitions = vector<vector<dim_t>>;
  _all_partitions_per_node = vector<_all_partitions>(dag.size());

  for(nid_t id = 0; id != dag.size(); ++id) {
    vector<vector<dim_t>> choices;
    for(auto const& dim: dag[id].dims) {
      choices.push_back(all_blocks(dim));
    }
    _all_partitions_per_node[id] = cartesian(
      choices,
      [this](vector<int> const& v)
      {
        return product(v) <= get_num_workers();
      });
    // sort it so that more workers come first
    std::sort(
      _all_partitions_per_node[id].begin(),
      _all_partitions_per_node[id].end(),
      [](vector<dim_t> const& lhs, vector<dim_t> const& rhs) {
        return product(lhs) > product(rhs);
      });
  }
}

vector<dim_t> const& PartitionOptions::get_which_partition(nid_t nid, int which) const {
  return all_partitions(nid)[which];
}

vector<dim_t> PartitionOptions::all_blocks(dim_t dim) const {
  vector<dim_t> ret;
  for(auto b: get_all_blocks()) {
    if(dim >= b && dim % b == 0 && b <= get_num_workers()) {
      ret.push_back(b);
    }
  }
  return ret;
}

vector<int> PartitionOptions::time_to_completion(vector<int> const& times) const {
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

vector<vector<dim_t>> PartitionOptions::get_random_full_dag_partitioning() const {
  random_select_t random_select(seed());
  auto f = [&](nid_t nid){ return random_select(this->all_partitions(nid)); };
  return _get_partitioning(f);
}

vector<vector<dim_t>> PartitionOptions::get_max_full_dag_partitioning() const {
  auto f = [&](nid_t nid){ return this->max_partition(nid); };
  return _get_partitioning(f);
}

vector<dim_t>
PartitionOptions::get_kernel_inc_dims(
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

vector<int> PartitionOptions::get_out(vector<int> const& xs, nid_t nid) const {
  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::join) {
    vector<int> ret;
    for(int which_inc = 0; which_inc != xs.size(); ++which_inc) {
      auto iter = std::find(node.aggs.begin(), node.aggs.end(), which_inc);
      bool is_agg_dim = iter != node.aggs.end();
      if(!is_agg_dim) {
        ret.push_back(xs[which_inc]);
      }
    }
    return ret;
  } else {
    return xs;
  }
}

vector<int> PartitionOptions::get_agg(vector<int> const& xs, nid_t nid) const {
  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::join) {
    vector<int> ret;
    for(int which_inc = 0; which_inc != xs.size(); ++which_inc) {
      auto iter = std::find(node.aggs.begin(), node.aggs.end(), which_inc);
      bool is_agg_dim = iter != node.aggs.end();
      if(is_agg_dim) {
        ret.push_back(xs[which_inc]);
      }
    }
    return ret;
  } else {
    return xs;
  }
}

vector<int> PartitionOptions::get_reblock_out(
  vector<int> const& join_inc,
  nid_t reblock_id) const
{
  node_t const& reblock_node = dag[reblock_id];
  node_t const& join_node = dag[reblock_node.ups[0]];

  int which_input = 0;
  for(; which_input != join_node.downs.size(); ++which_input) {
    if(join_node.downs[which_input] == reblock_id) {
      break;
    }
  }
  if(which_input == join_node.downs.size()) {
    throw std::runtime_error("up join node doesn't have down reblock");
  }

  auto const& ordering = join_node.ordering[which_input];
  vector<dim_t> ret;
  for(auto which_inc: ordering) {
    ret.push_back(join_inc[which_inc]);
  }
  return ret;
}

vector<int> PartitionOptions::get_out_from_compute(
  vector<int> const& inc,
  nid_t nid) const
{
  node_t const& node = dag[nid];
  if(node.type == node_t::node_type::reblock) {
    return get_reblock_out(inc, nid);
  }
  if(node.type == node_t::node_type::agg) {
    return get_out(inc, node.downs[0]);
  }
  if(node.type == node_t::node_type::join) {
    return get_out(inc, nid);
  }
  if(node.type == node_t::node_type::input) {
    return inc;
  }
  throw std::runtime_error("get_out_from_compute: invalid node type");
  return {};
}

tuple<int, int>
PartitionOptions::get_est_duration_worker_pair(
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

tuple<int, int> PartitionOptions::get_duration_worker_pair(
  vector<vector<int>> const& all_parts,
  nid_t nid) const
{
  return get_duration_worker_pair(
           [&all_parts](nid_t which) { return all_parts[which]; },
           nid);
}

tuple<int, int> PartitionOptions::get_duration_worker_pair(
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

void PartitionOptions::_super_depth_dag_order_add_to_ret(
  std::map<nid_t, int>& counts,
  vector<nid_t>& ret,
  nid_t id) const
{
  node_t const& node = dag[id];

  ret.push_back(id);

  // get the compute ups
  vector<nid_t> ups = get_compute_ups(id);

  for(nid_t const& up: ups) {
    counts.at(up)--; // do bounds checking, just in case.
    if(counts[up] == 0) {
      _super_depth_dag_order_add_to_ret(counts, ret, up);
    }
  }
}

void PartitionOptions::_depth_dag_order_add_to_ret(
  vector<nid_t>& counts,
  vector<nid_t>& ret,
  nid_t id) const
{
  ret.push_back(id);
  for(nid_t const& up: dag[id].ups) {
    counts[up]--;
    if(counts[up] == 0) {
      _depth_dag_order_add_to_ret(counts, ret, up);
    }
  }
}

vector<vector<dim_t>> PartitionOptions::_get_partitioning(
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

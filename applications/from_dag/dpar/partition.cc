#include "partition.h"

#include "../misc.h"

namespace bbts { namespace dag {

// TODO remove this
template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& xs)
{
  if(xs.size() == 0) {
    return os;
  }

  os << "[";
  for(int i = 0; i != xs.size() - 1; ++i) {
    os << xs[i] << ",";
  }
  os << xs.back() << "]";

  return os;
}


#define DCB01(x) std::cout << "dpar " << __LINE__ << " " << x << std::endl
#define DCB_SUPERIZE(x) std::cout << "dpar super " << __LINE__ << " " << x << std::endl

using std::set;
using std::unordered_map;

vector<vector<int>> run_partition(
  dag_t const& dag,
  search_params_t params,
  unordered_map<int, vector<int>> const& possible_parts)
{
  DCB01("enter run partition");
  solver_t solver(dag, possible_parts);

  for(nid_t nid: dag.breadth_dag_order()) {
    if(dag[nid].type == node_t::node_type::input) {
      // You can form a graph at an input node and solve it,
      // but it does not good since all inputs currently have zero cost.
      continue;
    }
    if(solver.can_solve_at(nid)) {
      // This will happen at every join node, basically. Regardless of
      // the depth at nid.. One could say only if the depth is deep enough
      // or it is an output node, solve.
      solver.solve(nid, params);
    }
  }

  DCB01("exit run partition");
  return solver.get_partition();
}

solver_t::solver_t(
  dag_t const& dag_,
  unordered_map<int, vector<int>> const& possible_parts_):
    dag(dag_),
    possible_parts(possible_parts_),
    partition(dag_.size())
{
  DCB01("sover_t constructor enter");
  // Initialize partition for join and input nodes to the first possible
  // part for that particular dimension size.
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.type == node_t::node_type::join ||
       node.type == node_t::node_type::input)
    {
      for(auto const& d: node.dims) {
        partition[nid].push_back(possible_parts.at(d)[0]);
      }
    }
  }

  // Setup the cost nodes
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.type == node_t::node_type::input) {
      cost_nodes.push_back(
        std::unique_ptr<cost_node_t>(new cost_node_t(nid)));
    } else
    if(node.type == node_t::node_type::join) {
      // Go through and figure out what the down nodes are.
      vector<nid_t> downs;
      downs.reserve(node.downs.size());
      // Since this is a join, by assumption all children all reblocks.
      // All childrren of each of those reblocks are either joins, inputs or aggs.
      // If it is an agg, get the join below it, otherwise it is an input or a join
      // which will be a valid cost node.
      for(nid_t const& reblock_nid: node.downs) {
        nid_t const& maybe_down_nid = dag[reblock_nid].downs[0];
        if(dag[maybe_down_nid].type == node_t::node_type::agg) {
          downs.push_back(dag[maybe_down_nid].downs[0]);
        } else {
          downs.push_back(maybe_down_nid);
        }
      }

      cost_nodes.push_back(
        std::unique_ptr<cost_node_t>(new cost_node_t(nid, downs)));
    } else {
      cost_nodes.push_back(nullptr);
    }
  }
  DCB01("sover_t constructor exit");
}

solver_t::coster_t::coster_t(nid_t top_nid, search_params_t params, solver_t* self):
  self(self), params(params)
{
  t_nids.insert_root(top_nid, get_options(top_nid));
  s_nids.insert(top_nid);

  if(params.max_depth > 1) {
    vector<tuple<nid_t, int>> so_far;
    so_far.emplace_back(top_nid, 1);
    for(int idx = 0; idx != so_far.size(); ++idx) {
      auto [nid, d] = so_far[idx];
      if(d < params.max_depth) {
        for(auto const& child: self->cost_nodes[nid]->downs) {
          t_nids.insert_child(nid, child, get_options(child));
          s_nids.insert(child);
          so_far.emplace_back(child, d+1);
        }
      }
    }
  }
  DCB01("COSTER has nids of " << std::vector<int>(s_nids.begin(), s_nids.end()));
}

void solver_t::solve(nid_t nid, search_params_t params) {
  using namespace std::placeholders;

  DCB01("solve A");

  // Set up the tree to solve with the dynamic programming algorithm
  coster_t coster(nid, params, this);

  DCB01("solve B");

  tree::f_cost_node_t<vector<int>> f_cost_node =
    std::bind(&solver_t::coster_t::cost_super_node, &coster, _1, _2);

  tree::f_cost_edge_t<vector<int>> f_cost_edge =
    std::bind(&solver_t::coster_t::cost_super_edge, &coster, _1, _2, _3, _4);

  // Find the best solution for this tree
  tree::tree_t<vector<int>> solved = tree::solve(
      coster.t_nids,
      f_cost_node,
      f_cost_edge);

  // Update the partition values here
  for(nid_t const& nid: coster.s_nids) {
    DCB01("solve found for " << nid << " a partition of " << solved[nid]);
    // TODO(wish, maybe):
    //   updating partition values should update the total cost, so that the
    //   solver maintains an accurate cost always
    partition[nid] = solved[nid];
  }

  DCB01("solve exit");
}

vector<vector<int>> solver_t::get_partition() const {
  vector<vector<int>> ret;
  ret.resize(dag.size());

  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.type == node_t::node_type::input) {
      ret[nid] = partition[nid];
    } else
    if(node.type == node_t::node_type::join) {
      ret[nid] = partition[nid];
    } else
    if(node.type == node_t::node_type::agg) {
      nid_t const& join_nid = node.downs[0];
      ret[nid] = dag.get_out(partition[join_nid], join_nid);
    } else
    if(node.type == node_t::node_type::reblock) {
      nid_t const& join_nid = node.ups[0];
      ret[nid] = dag.get_out_for_input(partition[join_nid], join_nid, nid);
    } else {
      throw std::runtime_error("should not reach");
    }
  }

  return ret;
}

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
void superize(vector<node_t>& dag) {
  DCB01("enter superize");
  for(nid_t join_nid = 0; join_nid != dag.size(); ++join_nid) {
    node_t& join_node = dag[join_nid];
    if(join_node.type != node_t::node_type::join) {
      continue;
    }
    DCB_SUPERIZE("on nid " << join_nid);

    for(nid_t& down_nid: dag[join_nid].downs) {
      node_t& down_node = dag[down_nid];
      if(down_node.type == node_t::node_type::reblock) {
        continue;
      }

      // Insert a reblock
      vector<int> reblock_dims =
        dag_t(dag).get_out_for_input(join_node.dims, join_nid, down_nid);

      nid_t reblock_nid = dag.size();
      dag.push_back(node_t {
        .type   = node_t::node_type::reblock,
        .id     = reblock_nid,
        .dims   = reblock_dims,
        .downs  = vector<nid_t>{down_nid},
        .ups    = vector<nid_t>{join_nid},
        .params = vector<param_t>{} // WARNING: Assuming reblocks never have parameters!
      });

      // join's down is the reblock now
      down_nid = reblock_nid;

      // down's up is the reblock now
      for(nid_t& to_middle_nid: down_node.ups) {
        if(to_middle_nid == join_nid) {
          to_middle_nid = reblock_nid;
        }
      }
    }
  }
  DCB01("exit superize");
}

vector<vector<int>> solver_t::coster_t::get_options(nid_t nid) {
  // TODO(necessary): add to params ways to limit the sie of options!!
  if(!params.all_parts) {
    throw std::runtime_error("get_options for all_parts = false not implemented");
  }
  // Just do the cartesian product.
  auto const& dims = self->dag[nid].dims;

  vector<vector<int>> os;
  os.reserve(dims.size());
  for(auto const& d: dims) {
    os.push_back(self->possible_parts.at(d));
  }

  return cartesian(os).resize(2);
}

// The cost of a "super" node is as follows:
//   1. The cost of the join or the input,
//   2. The aggregation if there is one,
//   3. Any reblocks from above that are not included in this coster
//   4. Any reblocks from below that are not included in this coster
uint64_t solver_t::coster_t::cost_super_node(
  nid_t nid, vector<int> const& partition) const
{
  auto const& dag = self->dag;
  auto const& node = dag[nid];

  // The cost of the join or input.
  uint64_t ret = cost_node(nid, partition);

  // The top of the super node and that nids partition
  nid_t super_top;
  vector<int> out_partition;

  // Is there an agg?
  if(node.ups.size() == 1 && dag[node.ups[0]].type == node_t::node_type::agg) {
    auto const& agg_nid = node.ups[0];

    // initialize the super info
    super_top = agg_nid;
    out_partition = dag.get_out(partition, nid);

    // add to the cost
    ret += cost_node(agg_nid, partition);
  } else {
    super_top = nid;
    out_partition = partition;
  }

  // Now get the compute nids of the up SUPER node... All up nodes will be reblocks.
  for(nid_t const& up_reblock_nid: dag[super_top].ups) {
    nid_t compute_nid = dag[up_reblock_nid].ups[0];

    // This node already belongs to this coster, so the
    // reblock cost will be computed by cost_edge.
    if(s_nids.count(compute_nid) > 0) {
      continue;
    }

    auto partition_up = dag.get_out_for_input(
      self->partition[compute_nid], compute_nid, up_reblock_nid);

    auto const& partition_down = out_partition;
    ret += cost_reblock(up_reblock_nid, partition_up, partition_down);
  }

  // Now any reblocks below this guy that are not included in the coster
  // have to be included
  for(nid_t const& reblock_nid: node.downs) {
    nid_t const& below_reblock_nid = dag[reblock_nid].downs[0];

    nid_t compute_nid;
    if(dag[below_reblock_nid].type == node_t::node_type::agg) {
      compute_nid = dag[below_reblock_nid].downs[0];
    } else {
      compute_nid = below_reblock_nid;
    }

    // This node already belongs to this coster, so the
    // reblock will be considered at the cost_edge
    if(s_nids.count(compute_nid) > 0) {
      continue;
    }

    auto partition_down = dag.get_out(
      self->partition[compute_nid], compute_nid);

    auto partition_up = dag.get_out_for_input(
      partition, nid, reblock_nid);

    ret += cost_reblock(reblock_nid, partition_up, partition_down);
  }

  return ret;
}

// The cost of an edge is the cost of the reblock between those edges
uint64_t solver_t::coster_t::cost_super_edge(
  nid_t parent_nid, vector<int> const& parent_partition,
  nid_t        nid, vector<int> const&        partition) const
{
  auto const& dag = self->dag;
  node_t const& node = dag[nid];

  nid_t super_top;
  vector<int> inn_partition;

  if(node.ups.size() == 1 && dag[node.ups[0]].type == node_t::node_type::agg) {
    auto const& agg_nid = node.ups[0];
    super_top = agg_nid;
    inn_partition = dag.get_out(partition, nid);
  } else {
    super_top = nid;
    inn_partition = partition;
  }

  bool found = false;
  for(nid_t const& up_reblock_nid: dag[super_top].ups) {
    nid_t compute_nid = dag[up_reblock_nid].ups[0];
    if(compute_nid == parent_nid) {
      auto upp_partition = dag.get_out_for_input(
        parent_partition, compute_nid, up_reblock_nid);

      return cost_reblock(up_reblock_nid, upp_partition, inn_partition);
    }
  }

  throw std::runtime_error("should not reach");
  return 0;
}

int _get_max_num_inputs_est(int inn, int out) {
  // Here is the 3 , 2 case
  //  inn xxxooo
  //  out xxooxx
  // In this case, the middle out partition depends
  // on both inputs. The goal is to figure out what the maximum
  // number of inputs could be...
  //
  // The idea is to place the out starting at the last of the first input
  // and then see how far it reaches
  //
  // inn 3 xxxoooxxxooo
  // out 5   ooooo
  // ^ the answer is 3

  // inn goes from [0,inn)
  // out goes from [inn-1, inn-1+out)
  int end = inn-1+out;
  return (end+inn-1) / inn; // celing(end / inn)
}

uint64_t solver_t::coster_t::cost_node(
  nid_t nid, vector<int> const& inc_partition) const
{
  auto const& dag = self->dag;

  if(dag[nid].type == node_t::node_type::input) {
    return 0;
  }

  if(dag[nid].type == node_t::node_type::agg) {
    auto const& agg_node = dag[nid];
    nid_t const& compute_nid = agg_node.downs[0];

    vector<int> agg_part = dag.get_agg(inc_partition,         compute_nid);
    uint64_t num_inn = 1;
    for(auto const& i: agg_part) {
      num_inn *= i;
    }
    if(num_inn == 1) {
      return 0;
    }

    vector<int> out_part = dag.get_out(inc_partition,         compute_nid);
    vector<int> out_dims = agg_node.dims;

    uint64_t bytes = 1;
    uint64_t num_out = 1;
    for(int i = 0; i != out_dims.size(); ++i) {
      bytes *= (out_dims[i] / out_part[i]);
      num_out *= out_part[i];
    }

    // Every input must be moved,
    // There are (num_inn - 1) additions.
    // Each output block can be added up in parallel.
    return _cost(num_out, num_inn * bytes, (num_inn - 1) * bytes, false);
  }

  if(dag[nid].type == node_t::node_type::join) {
    node_t const& node = dag[nid];

    uint64_t flops = 1;
    uint64_t num_parallel = 1;
    vector<int> local_inc_dims;
    local_inc_dims.resize(node.dims.size());
    for(int i = 0; i != node.dims.size(); ++i) {
      local_inc_dims.push_back(node.dims[i] / inc_partition[i]);
      flops *= local_inc_dims.back();
      num_parallel *= inc_partition[i];
    }

    uint64_t bytes = 0;
    for(nid_t const& down_nid: node.downs) {
      uint64_t p = 1;
      auto ds = dag.get_out_for_input(local_inc_dims, nid, down_nid);
      for(auto const& d: ds) {
        p *= d;
      }
      bytes += p;
    }

    return _cost(num_parallel, bytes, flops, false);
  }

  throw std::runtime_error("should not reach");
  return 0;
}

uint64_t solver_t::coster_t::cost_reblock(
  nid_t nid, vector<int> const& out_part, vector<int> const& inn_part) const
{
  if(inn_part == out_part) {
    return 0;
  }

  auto const& dims = self->dag[nid].dims;

  uint64_t inn_size;
  uint64_t num_out_blocks = 1;
  for(int i = 0; i != inn_part.size(); ++i) {
    inn_size *= (dims[i] / inn_part[i]);
    num_out_blocks *= out_part[i];
  }

  uint64_t max_num_inputs = 1;
  for(int r = 0; r != dims.size(); ++r) {
    max_num_inputs *= _get_max_num_inputs_est(inn_part[r], out_part[r]);
  }

  // The flops cost:
  //   every output block traverses (as a worst case) max_num_inputs * inn_size items.
  // The inputs cost:
  //   every output block needs     (as a worst case) max_num_inputs * inn_size items.
  uint64_t s = inn_size * max_num_inputs;
  return _cost(num_out_blocks, s, s, true);
}

inline uint64_t solver_t::coster_t::_cost(
  uint64_t num_parallel,
  uint64_t inn_bytes,
  uint64_t flops,
  bool is_reblock) const
{
  uint64_t reblock_multiplier = is_reblock ? params.reblock_multiplier : 1;

  uint64_t cost_per_item =
    (inn_bytes * (reblock_multiplier * params.inn_multiplier)) +
    (flops     * (reblock_multiplier * params.flops_multiplier));

  // If you have 13 (num_parallel) units of work to do,
  // and 6 (params.num_workers) workers on it, it will take
  // (ceiling [13 / 6] = 3) * however long one unit takes.
  uint64_t parallel_multiplier = (num_parallel + params.num_workers - 1) / params.num_workers;

  return parallel_multiplier * cost_per_item;
}

}}

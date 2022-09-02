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


#define DCB01(x)        // std::cout << "dpar " << __LINE__ << " " << x << std::endl
#define DCB_SUPERIZE(x) // std::cout << "dpar super " << __LINE__ << " " << x << std::endl
#define DCB_B(x) //std::cout << "batch prob " << __LINE__ << "| " << x << std::endl
#define DCB_X(x) //std::cout << x << std::endl

using std::set;
using std::unordered_map;

vector<vector<int>> run_partition(
  dag_t const& dag,
  search_params_t const& params,
  unordered_map<int, vector<int>> const& possible_parts)
{
  DCB01("enter run partition");
  solver_t solver(dag, params, possible_parts);

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
      solver.solve(nid);
    }
  }

  solver.params.include_outside_up_reblock = true;

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
      solver.solve(nid);
    }
  }

  //DCB01("exit run partition");
  return solver.get_partition();
}

solver_t::solver_t(
  dag_t const& dag_,
  search_params_t const& params_,
  unordered_map<int, vector<int>> const& possible_parts_):
    dag(dag_),
    params(params_),
    possible_parts(possible_parts_),
    partition(dag_.size())
{
  DCB01("sover_t constructor enter");

  // For each relation, compute the total bytes and flops size.
  relation_bytes.reserve(dag.size());
  relation_flops.reserve(dag.size());
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];

    relation_flops.push_back(product(node.dims));
    relation_bytes.push_back(product(dag.get_out(node.dims, nid)));
  }

  // Now scale the relations_x sizings to be in the windows proivded
  {
    uint64_t max_flops = *std::max_element(relation_flops.begin(), relation_flops.end());
    uint64_t max_bytes = *std::max_element(relation_bytes.begin(), relation_bytes.end());
    uint64_t rng_flops = params.flops_scale_max - params.flops_scale_min;
    uint64_t rng_bytes = params.bytes_scale_max - params.bytes_scale_min;
    for(nid_t i = 0; i != dag.size(); ++i) {
      relation_flops[i] = params.flops_scale_min + (relation_flops[i] * rng_flops) / max_flops;
      relation_bytes[i] = params.bytes_scale_min + (relation_bytes[i] * rng_bytes) / max_bytes;
    }
  }

  // For each possible part, sort it
  for(std::pair<const int, vector<int>>& pair: possible_parts) {
    std::sort(pair.second.begin(), pair.second.end());
  }

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

solver_t::coster_t::coster_t(nid_t top_nid, solver_t* self):
  self(self)
{
  auto const& params = self->params;

  t_nids.insert_root(top_nid, get_options(top_nid));
  s_nids.insert(top_nid);

  if(params.max_depth > 1) {
    vector<tuple<nid_t, int>> so_far;
    so_far.emplace_back(top_nid, 1);
    for(int idx = 0; idx != so_far.size(); ++idx) {
      auto [nid, d] = so_far[idx];
      if(d < params.max_depth) {
        for(auto const& child: self->cost_nodes[nid]->downs) {
          // There is the annoying case where child is already included
          // in this tree.
          if(s_nids.count(child) == 0) {
            t_nids.insert_child(nid, child, get_options(child));
            s_nids.insert(child);
            so_far.emplace_back(child, d+1);
          }
        }
      }
    }
  }
  DCB01("COSTER has nids of " << std::vector<int>(s_nids.begin(), s_nids.end()));
}

void solver_t::solve(nid_t nid) {
  using namespace std::placeholders;

  DCB01("solve A");

  // Set up the tree to solve with the dynamic programming algorithm
  coster_t coster(nid, this);

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
    if(node.type == node_t::node_type::reblock ||
       node.type == node_t::node_type::mergesplit)
    {
      nid_t const& join_nid = node.ups[0];
      ret[nid] = dag.get_out_for_input(partition[join_nid], join_nid, nid);
    } else {
      throw std::runtime_error("get_partition: should not reach");
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
    if(dag[join_nid].type != node_t::node_type::join) {
      continue;
    }

    // Since this function modifies dag throughout, and vector resizing invalidates
    // references, make sure a vector resize does not happen.
    dag.reserve(dag.size() + 2 * dag[join_nid].downs.size());

    // Now update and push into the graph in place.
    node_t& join_node = dag[join_nid];
    DCB_SUPERIZE("on nid " << join_nid);

    for(nid_t& down_nid: dag[join_nid].downs) {
      node_t& down_node = dag[down_nid];
      if(down_node.type == node_t::node_type::reblock ||
         down_node.type == node_t::node_type::mergesplit)
      {
        continue;
      }
      DCB_SUPERIZE("nid " << join_nid << " connected down to "
        << down_nid << " | down_node ups " << down_node.ups);

      // Insert a reblock
      vector<int> reblock_dims =
        dag_t(dag).get_out_for_input(join_node.dims, join_nid, down_nid);

      DCB_SUPERIZE("A" << down_node.ups);

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
      DCB_SUPERIZE("B" << down_node.ups);
      down_nid = reblock_nid;
      DCB_SUPERIZE("C" << down_node.ups);

      // down's up is the reblock now
      bool found_it = false;
      for(nid_t& to_middle_nid: down_node.ups) {
        DCB_SUPERIZE("TO_MIDDLE_NID " << to_middle_nid);
        if(to_middle_nid == join_nid) {
          found_it = true;
          to_middle_nid = reblock_nid;
          break;
        }
      }
      if(!found_it) { throw std::runtime_error("error in superize"); }
    }
  }
  DCB01("exit superize");
}

vector<vector<int>> solver_t::coster_t::get_options(nid_t nid) {
  auto const& params = self->params;
  if(params.all_parts) {
    // Just do the cartesian product.
    auto const& dims = self->dag[nid].dims;

    vector<vector<int>> os;
    os.reserve(dims.size());
    for(auto const& d: dims) {
      os.push_back(self->possible_parts.at(d));
    }

    return cartesian(os);
  } else {
    auto const& dims = self->dag[nid].dims;

    auto const& current = self->partition.at(nid);

    vector<vector<int>> os;
    os.reserve(dims.size());

    for(int i = 0; i != dims.size(); ++i) {
      // - Add 1 if it is in possible parts
      // - Add the current value
      // - Add the first value larger than the current value,
      //     if the current value isn't the largest

      auto const& c = current[i];
      auto const& d = dims[i];
      auto const& parts = self->possible_parts.at(d);

      vector<int> ps{c};
      if(parts[0] == 1) {
        ps.push_back(1);
      }
      auto w = std::upper_bound(parts.begin(), parts.end(), c);
      if(w != parts.end()) {
        ps.push_back(*w);
      }

      os.push_back(std::move(ps));
    }

    return cartesian(os);
  }
}

// The cost of a "super" node is as follows:
//   1. The cost of the join or the input,
//   2. The aggregation if there is one,
//   3. Any reblocks from above that are not included in this coster
//   4. Any reblocks from below that are not included in this coster
// For cases 3 and 4, if any of the up nodes or down nodes are already
// included in this tree, the reblock is just ignored.. Consider the
// following:
//           [a]
//         [b    c]
//        [x d] [x e]
// The problem is that to coster, the duplicate x will be removed:
//          [a]
//        [b   c]
//      [x d] [e]
// And during the dynamic programming problem, we can't tell what
// the c-x edge will reveal...
uint64_t solver_t::coster_t::cost_super_node(
  nid_t nid, vector<int> const& partition) const
{
  auto const& params = self->params;
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

  if(params.include_outside_up_reblock) {
    // Now get the compute nids of the up SUPER node... All up nodes will be reblocks.
    for(nid_t const& up_reblock_nid: dag[super_top].ups) {
      nid_t compute_nid = dag[up_reblock_nid].ups[0];

      // This node already belongs to this coster, so the
      // reblock cost will be computed by cost_edge or it has to
      // be ignored.
      if(s_nids.count(compute_nid) > 0) {
        continue;
      }

      auto partition_up = dag.get_out_for_input(
        self->partition[compute_nid], compute_nid, up_reblock_nid);

      auto const& partition_down = out_partition;
      ret += cost_reblock_or_mergesplit(up_reblock_nid, partition_up, partition_down);
    }
  }

  if(params.include_outside_down_reblock) {
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
      // reblock will be considered at the cost_edge or it
      // has to be ignored all together
      if(s_nids.count(compute_nid) > 0) {
        continue;
      }

      auto partition_down = dag.get_out(
        self->partition[compute_nid], compute_nid);

      auto partition_up = dag.get_out_for_input(
        partition, nid, reblock_nid);

      ret += cost_reblock_or_mergesplit(reblock_nid, partition_up, partition_down);
    }
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

  for(nid_t const& up_reblock_nid: dag[super_top].ups) {
    if(dag[up_reblock_nid].type != node_t::node_type::reblock &&
       dag[up_reblock_nid].type != node_t::node_type::mergesplit)
    {
      throw std::runtime_error("invalid up reblock or mergesplit nid!!!");
    }

    nid_t compute_nid = dag[up_reblock_nid].ups[0];
    if(compute_nid == parent_nid) {
      auto upp_partition = dag.get_out_for_input(
        parent_partition, compute_nid, up_reblock_nid);

      return cost_reblock_or_mergesplit(up_reblock_nid, upp_partition, inn_partition);
    }
  }

  throw std::runtime_error("cost_super_edge: should not reach");
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

// There are two values of a relation: relation_bytes, relation_flops.
// Let T = the relation_flops + the number of incoming bytes
// Then:
//   T * w: the total relation cost
//   n    : the number of blocks in this partitioning
//   w    : the number of workers
//   p    : parallel multiplier
//   C    : cost associated with this partitioning
//   c    : cost per block
// The actual, final cost associated with this partitioning is
//   C := p * c
// The cost per/block is:
//   Tw / min(n, w).
// The parallel multiplier is:
//   p = ceiling( n / w )
// Example: w = 4
//   n  c   p  C
//   1  4T  1  4T
//   2  2T  1  2T
//   4   T  1   T
//   8   T  2  2T
// The implication:
//   The goldilocks zone is to be partitioned at the number of workers.
//   There is a penalty for less and a penalty for more.
inline uint64_t solver_t::coster_t::_cost(uint64_t total, int num_parallel) const {
  auto const& params = self->params;
  auto const& num_workers = params.num_workers;

  uint64_t cost_per_block =
    (total * params.num_workers) / std::min(num_parallel, num_workers);

  uint64_t parallel_multiplier = (num_parallel + num_workers - 1) / num_workers;

  return cost_per_block * parallel_multiplier;
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

    int num_agg = product(dag.get_agg(inc_partition, compute_nid));
    if(num_agg == 1) {
      return 0;
    }
    int num_blk = product(dag.get_out(inc_partition, compute_nid));

    // Each block has num_agg inputs
    uint64_t total_bytes = self->relation_bytes[nid] * num_agg;

    // num_agg blocks to agg -> num_agg-1 additions, num_blocks times.
    uint64_t total_flops = self->relation_flops[nid] * (num_agg-1);

    return _cost(total_bytes + total_flops, num_blk);
  }

  if(dag[nid].type == node_t::node_type::join) {
    node_t const& node = dag[nid];

    int num_blk = product(inc_partition);
    int num_agg = product(dag.get_agg(inc_partition, nid));

    int mult;

    // Each block has an input from each input relation
    uint64_t total_bytes = 0;
    uint64_t max_inn_bytes = 0;
    for(nid_t const& down_nid: node.downs) {
      total_bytes += self->relation_bytes[down_nid];
      if(self->relation_bytes[down_nid] > max_inn_bytes) {
        max_inn_bytes = self->relation_bytes[down_nid];
	mult = product(dag.get_out_for_input(inc_partition, nid, down_nid)) / num_agg;
      }
    }
    // We don't "move" the largest relation
    total_bytes -= max_inn_bytes;

    // But we broadcast to the largest location
    //total_bytes *= mult;

    // This relation has this many flops
    uint64_t const& total_flops = self->relation_flops[nid];

    return _cost(total_bytes + total_flops, num_blk);
  }

  throw std::runtime_error("cost_node: should not reach");
  return 0;
}

uint64_t solver_t::coster_t::cost_reblock_or_mergesplit(
  nid_t up_reblock_nid, vector<int> const& out_part, vector<int> const& inn_part) const
{
  auto const& dag = self->dag;

  if(dag[up_reblock_nid].type == node_t::node_type::mergesplit)
  {
    return cost_mergesplit(up_reblock_nid, out_part, inn_part);
  }
  if(dag[up_reblock_nid].type == node_t::node_type::reblock)
  {
    return cost_reblock(   up_reblock_nid, out_part, inn_part);
  }
  throw std::runtime_error("must have either a reblock or a mergesplit node");
  return 0;
}

uint64_t solver_t::coster_t::cost_mergesplit(
  nid_t nid, vector<int> const& out_part, vector<int> const& inn_part) const
{
  // Find the equivalent reblock and just call cost_reblock

  vector<int> inn_part_fixed;
  vector<int> out_part_fixed;

  auto const& node = self->dag[nid];
  if(node.is_merge) {
    // IJij -> Kk (= 1K1k)
    inn_part_fixed = inn_part;
    out_part_fixed = expand1(out_part);
  } else {
    // Kk (= 1K1k) -> IJij
    inn_part_fixed = expand1(inn_part);
    out_part_fixed = out_part;
  }

  return this->cost_reblock(nid, out_part_fixed, inn_part_fixed);
}

uint64_t solver_t::coster_t::cost_reblock(
  nid_t nid, vector<int> const& out_part, vector<int> const& inn_part) const
{
  DCB_B("cost_reblock at " << nid << ". inn,out " << inn_part << ", " << out_part);

  auto const& params = self->params;

  if(inn_part == out_part) {
    return 0;
  }

  int num_blk = product(out_part);

  uint64_t max_num_inputs = 1;
  for(int r = 0; r != inn_part.size(); ++r) {
    max_num_inputs *= _get_max_num_inputs_est(inn_part[r], out_part[r]);
  }

  uint64_t const& num_bytes = self->relation_bytes[nid] * max_num_inputs;
  uint64_t const& num_flops = self->relation_flops[nid] * max_num_inputs;

  // It may not really be a barirer, but this'll do
  bool is_barrier = false;
  for(int i = 0; i != inn_part.size(); ++i) {
    if(out_part[i] < inn_part[i]) {
      is_barrier = true;
      break;
    }
  }
  //int barrier_multiplier = is_barrier
  //      ? params.barrier_reblock_multiplier
  //      : 1
  //      ;

  DCB_B("case: reblock");

  uint64_t intercept = params.reblock_intercept +
		(is_barrier ? params.barrier_reblock_intercept : 0);
  return intercept + _cost(num_bytes + num_flops, num_blk);

  //return barrier_multiplier * params.reblock_multiplier *
  //         _cost(num_bytes + num_flops, num_blk);
}

}}

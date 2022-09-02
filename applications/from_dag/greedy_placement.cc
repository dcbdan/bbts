#include "greedy_placement.h"
#include "dpar/min_cost_tree.h"

#define DCB01(x) // std::cout << __LINE__ << " " << x << std::endl;


namespace bbts { namespace dag {

struct compute_score_t {
  compute_score_t(
    bool with_outputs,
    int num_ranks,
    relations_t const& relations,
    vector<placement_t> const& placements):
      with_outputs(with_outputs),
      num_ranks(num_ranks),
      relations(relations),
      placements(placements)
  {}

  // Reach past dummy joins
  vector<tuple<nid_t, int>> get_useful_outputs(nid_t an_nid, int an_idx) const
  {
    auto const& dag = relations[0].dag;

    auto const& immediate_outputs = relations[an_nid].get_outputs(an_idx);

    vector<tuple<nid_t, int>> ret;
    ret.reserve(2*immediate_outputs.size());
    for(auto const& [nid, idx]: immediate_outputs) {
      // If this is a join with one input, recursively get it's outputs
      if(dag[nid].downs.size() == 1 && dag[nid].type == node_t::node_type::join) {
        auto add_these = get_useful_outputs(nid, idx);
        for(auto const& x: add_these) {
          ret.push_back(x);
        }
        continue;
      }

      // This is not a dummy join, so it is useful
      ret.emplace_back(nid, idx);
    }

    return ret;
  }

  // Reach past dummy joins
  vector<tuple<nid_t, int>> get_useful_inputs(nid_t an_nid, int an_idx) const
  {
    auto const& dag = relations[0].dag;

    auto const& immediate_inputs = relations[an_nid].get_inputs(an_idx);

    vector<tuple<nid_t, int>> ret;
    ret.reserve(immediate_inputs.size());
    for(auto const& [nid, idx]: immediate_inputs) {
      // If this is a join with one input, recursively get it's inputs
      if(dag[nid].downs.size() == 1 && dag[nid].type == node_t::node_type::join) {
        auto useful_input = get_useful_inputs(nid, idx);
        if(useful_input.size() != 1) {
          throw std::runtime_error("!");
        }
        ret.push_back(useful_input[0]);
        continue;
      }

      // This is not a dummy join, so it is useful
      ret.emplace_back(nid, idx);
    }

    return ret;
  }

  // For an output (an_nid, an_idx), prefer_locs(an_nid, an_idx) gives
  // the set of locations where assigning an additional input is cost free.
  //
  // For touch ops:
  //   the locations that share the most number of inputs is preferred
  // For reduce ops:
  //   the locations that already have an input are preferred
  // For apply ops:
  //   the locations that share the most number of inputs is preferred
  std::set<int> prefer_locs(nid_t an_nid, int an_idx) const {
    auto const& dag = relations[0].dag;

    vector<int> counts(num_ranks, 0);

    auto inputs = get_useful_inputs(an_nid, an_idx);
    int num_assigned = 0;
    for(auto const& [nid, idx]: inputs) {
      // If there is a compute location for this input, then increment counts.
      if(placements[nid].size() > 0 && placements[nid].computes[idx] != -1) {
        int const& loc = placements[nid].computes[idx];
        counts[loc]++;
        num_assigned++;
      }
    }

    if(dag[an_nid].type == node_t::node_type::agg) {
      // This is a reduce op!
      if(num_assigned == 0) {
        // Nothing was assigned, there is no cost with any node
        std::set<int> ret;
        for(int loc = 0; loc != num_ranks; ++loc) {
          ret.insert(loc);
        }
        return ret;
      } else {
        std::set<int> ret;
        for(int loc = 0; loc != num_ranks; ++loc) {
          if(counts[loc] > 0) {
            ret.insert(loc);
          }
        }
        return ret;
      }
    } else {
      std::set<int> ret;

      int most = *std::max_element(counts.begin(), counts.end());

      for(int loc = 0; loc != num_ranks; ++loc) {
        if(counts[loc] == most) {
          ret.insert(loc);
        }
      }

      return ret;
    }
  }

  vector<uint64_t> operator()(
    nid_t an_nid,
    int   an_idx) const
  {
    vector<uint64_t> ret(num_ranks, 0);

    auto const& inputs = relations[an_nid].get_inputs(an_idx);

    for(auto const& [nid, idx]: inputs) {
      auto const& current_locs = placements[nid].locs[idx];
      for(int rank = 0; rank != num_ranks; ++rank) {
        if(current_locs.count(rank) == 0) {
          // increment by the size of the input tensor
          ret[rank] += relations[nid].tensor_size();
        }
      }
    }

    if(with_outputs) {
      auto outputs = get_useful_outputs(an_nid, an_idx);
      for(auto const& [nid, idx]: outputs) {
        auto cost_free_locs = prefer_locs(nid, idx);
        for(int rank = 0; rank != num_ranks; ++rank) {
          if(cost_free_locs.count(rank) == 0) {
            // increment by the size of the current tensor under consideration
            ret[rank] += relations[an_nid].tensor_size();
          }
        }
      }
    }

    return ret;
  }

private:
  bool with_outputs;
  int const num_ranks;

  // This is truly constant, and the reference won't be modified throughout this lifetime.
  relations_t const& relations;

  // This is only constant here, the reference will be modified throught this lifetime.
  vector<placement_t> const& placements;
};

// 1. Every relation must be node balanced
// 2. Input relations are round robin assigned
// 3. For each non-input relation:
//      For each block in that relation,
//         pick the cheapest rank among the non-full ranks,
//         update the necessary moves,
//         update the loc counter
// Also note:
// - reduces and touches (aggs and reblocks) do not add to input locs
//     > touches compact before move
//     > reduces local agg before move
vector<placement_t> greedy_solve_placement(
  bool with_order,
  bool with_outputs,
  relations_t const& relations,
  int num_nodes)
{
  DCB01("A");

  auto const& dag = relations[0].dag;

  // Start offset_rank at 1 since rank 0 is the master node
  // that issues commands to the other nodes.
  int offset_rank = num_nodes > 1 ? 1 : 0;

  auto increment_offset_rank = [&]() {
    offset_rank++;
    if(offset_rank == num_nodes) {
      offset_rank = 0;
    }
  };

  // Initialize the returned value
  vector<placement_t> ret;
  ret.resize(relations.size());
  // A relation with zero inputs is either (1) a no op or (2) not yet computed

  DCB01("B");

  // Round robin assign all inputs
  for(nid_t nid = 0; nid != relations.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.downs.size() == 0) {
      size_t num_blocks = relations[nid].get_num_blocks();

      ret[nid] = placement_t(num_blocks);

      for(int i = 0; i != ret[nid].computes.size(); ++i) {
        ret[nid].locs[i].insert(offset_rank);
        ret[nid].computes[i] = offset_rank;

        increment_offset_rank();
      }
    }
  }

  DCB01("C");

  compute_score_t compute_scores(with_outputs, num_nodes, relations, ret);

  DCB01("D");

  // For each non input, compute relation, assing the compute locations
  // and update where each block ends up being moved to
  for(nid_t const& nid: dag.breadth_dag_order()) {
    // Skip no ops and input nodes
    if(relations[nid].is_no_op() || dag[nid].downs.size() == 0) {
      continue;
    }

    int num_blocks = relations[nid].get_num_blocks();

    ret[nid] = placement_t(num_blocks);

    if(dag[nid].downs.size() == 1 && dag[nid].type == node_t::node_type::join) {
      // If this is a join with one input, each block should end up happening in the same
      // place as the input block.
      //
      // It will guaranteed to be load balanced in the same way as the input relation,
      // by properties of joins.

      for(int idx = 0; idx != num_blocks; ++idx) {
        // Note that input_nid will be the same for each idx
        // but dag[nid].downs[0] may be a no op and therefore not equal
        // to input_nid.
        auto const& inputs = relations[nid].get_inputs(idx);
        auto [input_nid, input_idx] = inputs[0];

        int loc = ret[input_nid].computes[input_idx];

        ret[nid].computes[idx] = loc;
        ret[nid].locs[idx].insert(loc);

      }

      continue;
    }

    int min_per_rank = num_blocks / num_nodes;

    vector<int> counts;

    // Use the offset rank to determine how many blocks
    // will be done by each rank
    counts = vector<int>(num_nodes, min_per_rank);
    int rem = num_blocks % num_nodes;
    for(int i = 0; i != rem; ++i) {
      counts[offset_rank]++;
      increment_offset_rank();
    }

    auto pick_best_score = [&](vector<uint64_t> const& scores) -> tuple<int,int>
    {
      int best_rank = -1;
      uint64_t best_score;
      for(int rank = 0; rank != num_nodes; ++rank) {
        uint64_t const& score = scores[rank];
        if(counts[rank] > 0) {
          if(best_rank < 0 || score < best_score) {
            best_rank = rank;
            best_score = score;
          }
        }
      }
      return tuple<int,uint64_t>(best_rank, best_score);
    };

    auto update = [&](int idx, int rank) {
      // The compute location has been chosen, update the information

      // set compute
      ret[nid].computes[idx] = rank;
      ret[nid].locs[idx].insert(rank);

      // update counts
      counts[rank]--;

      // update the input ranks for tensors that got moved..
      // reblock and agg don't count because
      // touch   and reduce don't move the input tensors
      if(dag[nid].type == node_t::node_type::join) {
        auto const& inputs = relations[nid].get_inputs(idx);
        for(auto const& [input_nid, input_idx]: inputs) {
          ret[input_nid].locs[input_idx].insert(rank);
        }
      }
    };

    if(with_order) {
      std::set<int> blocks;
      for(int idx = 0; idx != num_blocks; ++idx) {
        blocks.insert(idx);
      }

      auto pick_best_block = [&]() -> tuple<int,int>
      {
        int best_idx = -1;
        int best_rank;
        uint64_t best_score;
        for(int idx: blocks) {
          auto scores = compute_scores(nid, idx);
          auto [rank, score] = pick_best_score(scores);
          if(best_idx < 0 || score < best_score) {
            best_idx   = idx;
            best_rank  = rank;
            best_score = score;
          }
        }
        return tuple<int,int>(best_idx, best_rank);
      };


      while(blocks.size() != 0) {
        auto [idx, best_rank] = pick_best_block();
        blocks.erase(idx);
        update(idx, best_rank);
      }
    } else {
      for(int idx = 0; idx != num_blocks; ++idx) {
        auto scores = compute_scores(nid, idx); // uses ret
        auto [best_rank, best_score] = pick_best_score(scores);
        update(idx, best_rank);
      }
    }
  }

  DCB01("E");

  return ret;
}

uint64_t total_move_cost(
  relations_t const& relations,
  vector<vector<int>> const& computes)
{
  auto const& dag = relations[0].dag;

  vector<placement_t> placements;
  for(auto const& x: computes) {
    placements.emplace_back(x);
  }

  uint64_t total = 0;

  // make sure that this block lives at rank
  auto assure_moved_to = [&](nid_t nid, int idx, int rank) {
    if(placements[nid].locs[idx].count(rank) == 0) {
      // Touches don't actually move the whole tensor,
      // they move some part of the tensor.
      if(dag[nid].type != node_t::node_type::reblock &&
         dag[nid].type != node_t::node_type::mergesplit)
      {
        placements[nid].locs[idx].insert(rank);
      }
      total += relations[nid].tensor_size();
    }
  };

  auto move_to_reblock = [&](nid_t an_nid, vector<int> const& an_bid) {
    int an_idx = relations[an_nid].bid_to_idx(an_bid);

    int rank = placements[an_nid].computes[an_idx];

    auto input_sizes = relations[an_nid].input_tensor_sizes(an_bid);
    auto inputs = relations[an_nid].get_inputs(an_idx);
    for(int i = 0; i != inputs.size(); ++i) {
      auto const& [input_nid, input_idx] = inputs[i];
      auto const& input_size = input_sizes[i];
      if(placements[input_nid].locs[input_idx].count(rank) == 0) {
        total += input_size;
      }
    }
  };

  // Given a reduce block, move the locally aggregated inputs to it
  auto move_to_reduce = [&](nid_t an_nid, int an_idx) {
    int reduce_rank = placements[an_nid].computes[an_idx];

    std::set<int> input_ranks;
    auto const& inputs = relations[an_nid].get_inputs(an_idx);
    for(auto const& [input_nid, input_idx]: inputs) {
      auto other_rank = placements[input_nid].computes[input_idx];
      if(other_rank != reduce_rank) {
        input_ranks.insert(other_rank);
      }
    }

    auto const& [input_nid, _0] = inputs[0];
    total += input_ranks.size() * relations[input_nid].tensor_size();
  };

  // For each relation that is not a no op and not an input:
  //   For each block in that relation:
  //     do the corresponding op to increment total
  // return total

  for(nid_t const& nid: dag.breadth_dag_order()) {
    // Skip no ops and input nodes
    if(relations[nid].is_no_op() || dag[nid].downs.size() == 0) {
      continue;
    }

    //int num_blocks = relations[nid].get_num_blocks();
    //for(int idx = 0; idx != num_blocks; ++idx)

    indexer_t indexer(relations[nid].partition);
    do {
      auto const& bid = indexer.idx;
      int idx = relations[nid].bid_to_idx(bid);

      if(dag[nid].type == node_t::node_type::agg) {
        move_to_reduce(nid, idx);
      } else
      if(dag[nid].type == node_t::node_type::reblock ||
         dag[nid].type == node_t::node_type::mergesplit)
      {
        move_to_reblock(nid, bid);
      } else {
        int rank = placements[nid].computes[idx];
        auto const& inputs = relations[nid].get_inputs(idx);
        for(auto const& [input_nid, input_idx]: inputs) {
          assure_moved_to(input_nid, input_idx, rank);
        }
      }
    } while(indexer.increment());
  }

  return total;
}

vector<vector<int>> just_computes(vector<placement_t> const& placements)
{
  vector<vector<int>> ret;
  ret.reserve(placements.size());
  for(placement_t const& item: placements) {
    if(item.computes.size() > 0) {
      ret.push_back(item.computes);
    } else {
      ret.push_back(vector<int>());
    }
  }
  return ret;
}

vector<vector<int>> dumb_solve_placement(
  relations_t const& relations,
  int num_nodes)
{
  auto const& dag = relations[0].dag;

  // Start offset_rank at 1 since rank 0 is the master node
  // that issues commands to the other nodes.
  int offset_rank = num_nodes > 1 ? 1 : 0;

  auto increment_offset_rank = [&]() {
    offset_rank++;
    if(offset_rank == num_nodes) {
      offset_rank = 0;
    }
  };

  // Initialize the returned value
  vector<vector<int>> ret;
  ret.resize(relations.size());
  // A relation with zero inputs is either (1) a no op or (2) not yet computed

  // Round robin assign all inputs
  for(nid_t nid = 0; nid != relations.size(); ++nid) {
    node_t const& node = dag[nid];
    if(!relations[nid].is_no_op()) {
      size_t num_blocks = relations[nid].get_num_blocks();

      ret[nid] = vector<int>(num_blocks);

      for(int i = 0; i != num_blocks; ++i) {
        ret[nid][i] = offset_rank;

        increment_offset_rank();
      }
    }
  }
  return ret;
}

struct dyn_solver_t {
  dyn_solver_t(
    relations_t const& relations,
    int num_nodes,
    vector<placement_t>& placements,
    vector<vector<int>>& load_limits):
      relations(relations), num_nodes(num_nodes),
      placements(placements), load_limits(load_limits)
  {}

  // Reach past dummy joins
  vector<tuple<nid_t, int>> get_useful_inputs(nid_t an_nid, int an_idx) const;
  void solve_and_update(nid_t nid, int bid) {
    tree_t t(nid, bid, this);
    t.solve_and_update();
  }

  void solve_and_update(nid_t nid);

  inline bool has_at(nid_t const& nid, int const& bid, int const& rank) const {
    return placements[nid].locs[bid].count(rank) > 0;
  }

  inline vector<int> options(nid_t const& nid, int const& bid) const {
    vector<int> ret;
    ret.reserve(num_nodes);
    for(int rank = 0; rank != num_nodes; ++rank) {
      if(load_limits[nid][rank] > 0) {
        ret.push_back(rank);
      }
    }
    if(ret.size() == 0) {
      throw std::runtime_error("could not find any options!");
    }
    return ret;
  }

  inline bool is_set(nid_t const& nid, int const& bid) const {
    return placements[nid].locs[bid].size() > 0;
  }

  struct tree_t {
    tree_t(nid_t nid, int id, dyn_solver_t* self);

    bool insert(int parent_idx, nid_t nid, int id);

    uint64_t cost_node(int idx,    int const& rank) const;
    uint64_t cost_edge(int idx_up, int const& rank_up,
                       int idx_dw, int const& rank_dw) const;

    void solve_and_update();

    tree::tree_t<vector<int>> options;

    vector<vector<tuple<nid_t, int, bool>>> idx_to_children;

    vector<tuple<nid_t, int>> idx_to_id;

    dyn_solver_t* self;
  };

  relations_t const& relations;
  int num_nodes;
  vector<placement_t>& placements;
  vector<vector<int>>& load_limits;
};

// 1. Input releations are round robin assigned
// 2. For each chosen relation
//      For each block in that relation,
//      - setup the tree to solve stemming from that block
//      - solve the dyn progrmaming problem over that tree
//      - update the placements and the load balancing count
// Also note:
// - reduces and touches do not add to input locs
//     > touches compact before move
//     > reduces local agg before move
// - the node cost in the dyn programming tree is the
//   moves from inputs that are already set
// - Which nodes to solve the dyn prog problem?
//   - All agg nodes. These guys reach past their joins and reblocks
//   - All joins with > 1 input and no agg above get chosen as well
vector<placement_t> dyn_solve_placement(
  relations_t const& relations,
  int num_nodes)
{
  auto const& dag = relations[0].dag;

  // Initialize the returned value
  vector<placement_t> placements(relations.size());

  // Start offset_rank at 1 since rank 0 is the master node
  // that issues commands to the other nodes.
  int offset_rank = num_nodes > 1 ? 1 : 0;

  auto increment_offset_rank = [&]() {
    offset_rank++;
    if(offset_rank == num_nodes) {
      offset_rank = 0;
    }
  };


  // Round robin assign all inputs
  for(nid_t nid = 0; nid != relations.size(); ++nid) {
    size_t num_blocks = relations[nid].get_num_blocks();
    node_t const& node = dag[nid];
    if(node.downs.size() == 0) {

      placements[nid] = placement_t(num_blocks);

      for(int i = 0; i != placements[nid].computes.size(); ++i) {
        placements[nid].locs[i].insert(offset_rank);
        placements[nid].computes[i] = offset_rank;

        increment_offset_rank();
      }
    } else {
      placements[nid] = placement_t(num_blocks);
    }
  }

  vector<vector<int>> load_limits(dag.size());
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.type == node_t::node_type::input) {
      continue;
    }

    int num_blocks = relations[nid].get_num_blocks();

    // Use ceil [ num_blocks / num_nodes ]

    // This might be problematic. Consider:
    //   num_blocks = 11, num_nodes = 10. Each block can get
    //   2 items. If the first five nodes get 2 items, the sixth 1 item,
    //   4 nodes don't do anything.
    // However, this will lead to lower move costs, which is important when
    // there are lots of nodes. Of course, placing everything on node 0
    // has the lowest move cost of all!

    // TODO(wish): add option to use increment_offset_rank and have a strict
    //             number on each node

    int max_per_rank = (num_blocks + num_nodes - 1) / num_nodes;
    load_limits[nid] = vector<int>(num_nodes, max_per_rank);
  }

  // This guy holds references to everything here and can solve from any node
  // requested. Even though it can solve from any node, the load balancing can
  // be violated.
  // TODO(wish): the dyn prog problem doesn't allow unbalanced solutions.
  // It also reaches past dummy joins and doesn't set them.
  dyn_solver_t solver(relations, num_nodes, placements, load_limits);

  // For each non input, compute relation, assing the compute locations
  // and update where each block ends up being moved to
  for(nid_t const& nid: dag.breadth_dag_order()) {
    // Skip no ops, input nodes and nodes that are already assigned
    if(relations[nid].is_no_op()  ||
       dag[nid].downs.size() == 0 ||
       placements[nid].locs[0].size() > 0)
    {
      continue;
    }

    // This is an unassigned node, but is it chosen?
    //   reblock or mergesplit           not chosen
    //   agg                             chosen
    //   join
    //     paired with agg               not chosen
    //     is "by-pass-join" or dummy    not chosen
    //     otherwise                     chosen

    node_t const& node = dag[nid];

    // Reblock nodes are never assigned directly
    if(node.type == node_t::node_type::reblock ||
       node.type == node_t::node_type::mergesplit)
    {
      continue;
    }
    // If this is a join paired with an agg THAT IS NOT A NO OP, solve at the agg so pass
    if(node.type == node_t::node_type::join &&
       node.ups.size() == 1 &&
       dag[node.ups[0]].type == node_t::node_type::agg &&
       !relations[node.ups[0]].is_no_op())
    {
      continue;
    }
    // If this is a join, the up node (that is computed) is not an agg.
    // If there is only 1 input, it is a by-pass-join, so do not choose it
    if(node.type == node_t::node_type::join && node.downs.size() == 1)
    {
      continue;
    }

    // We either have an agg or a join with more than one input and it should
    // be solved for.

    // solve stemming from this nid and update load_limits and placements
    solver.solve_and_update(nid);
  }

  // At this point every single placement is set _except_ dummy joins.
  // So set them here

  auto is_dummy = [&](nid_t nid) {
    return dag[nid].type == node_t::node_type::join &&
           (dag[nid].downs.size() == 1);
  };

  auto set_dummy = [&](nid_t _maybe_add) {
    std::vector<nid_t> stack;
    stack.reserve(3);

    // "recurse" by adding the nids that need to be operated onto the "stack"
    while(is_dummy(_maybe_add) && placements[_maybe_add].computes[0] == -1) {
      stack.push_back(_maybe_add);
      _maybe_add = dag[_maybe_add].downs[0];
    }

    // unwind the "stack"
    while(stack.size() != 0) {
      nid_t nid = stack.back();
      stack.pop_back();

      // update this dummy join
      for(int bid = 0; bid != relations[nid].get_num_blocks(); ++bid) {
        auto const& [down_nid, down_bid] = relations[nid].get_inputs(bid)[0];
        placements[nid].computes[bid] = placements[down_nid].computes[down_bid];
      }
    }
  };

  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    set_dummy(nid);
  }

  // At this point, every bid in placements should be >= 0.
  // TODO: remove this
  for(nid_t nid = 0; nid != placements.size(); ++nid) {
    auto const& placement = placements[nid];
    for(auto const& compute_rank: placement.computes) {
      if(compute_rank < 0 || compute_rank >= num_nodes) {
        throw std::runtime_error("dyn_solve_placement did not account for all compute locs!");
      }
    }
  }

  return placements;
}

void dyn_solver_t::solve_and_update(nid_t nid)
{
  // TODO(wish): What is the correct order to traverse the blocks?

  auto num_blocks = relations[nid].get_num_blocks();
  for(int idx = 0; idx != num_blocks; ++idx) {
    solve_and_update(nid, idx);
  }
}

// TODO(cleanup): this is just a copy paste of compute_score_t's get_useful_input
vector<tuple<nid_t, int>> dyn_solver_t::get_useful_inputs(nid_t an_nid, int an_idx) const
{
  auto const& dag = relations[0].dag;

  // This reaches past no ops
  auto const& immediate_inputs = relations[an_nid].get_inputs(an_idx);

  vector<tuple<nid_t, int>> ret;
  ret.reserve(immediate_inputs.size());
  for(auto const& [nid, idx]: immediate_inputs) {
    // If this is a join with one input, recursively get it's inputs
    if(dag[nid].downs.size() == 1 && dag[nid].type == node_t::node_type::join) {
      auto useful_input = get_useful_inputs(nid, idx);
      if(useful_input.size() != 1) {
        throw std::runtime_error("!");
      }
      ret.push_back(useful_input[0]);
      continue;
    }

    // This is not a dummy join, so it is useful
    ret.emplace_back(nid, idx);
  }

  return ret;
}

dyn_solver_t::tree_t::tree_t(nid_t nid, int id, dyn_solver_t* self_): self(self_)
{
  if(self->is_set(nid, id)) {
    throw std::runtime_error("very bad, this tree would have nothing in it!");
  }
  // just a guess on an upper bound size
  idx_to_id.reserve(20);
  idx_to_children.reserve(20);

  auto children = self->get_useful_inputs(nid, id);

  idx_to_id.emplace_back(nid, id);

  idx_to_children.push_back(vector<tuple<nid_t, int, bool>>());
  idx_to_children.back().reserve(children.size());

  options.insert_root(0, self->options(nid, id));
  for(auto const& [down_nid, down_id]: children) {
    // recursively build the tree
    bool did_add = insert(0, down_nid, down_id);
    idx_to_children[0].emplace_back(down_nid, down_id, did_add);
  }
}

bool dyn_solver_t::tree_t::insert(int parent_idx, nid_t nid, int id) {
  // This node doesn't belong in the tree since it has
  // been set.
  if(self->is_set(nid, id)) {
    return false;
  }

  int idx = idx_to_id.size();

  auto children = self->get_useful_inputs(nid, id);

  idx_to_id.emplace_back(nid, id);

  idx_to_children.push_back(vector<tuple<nid_t, int, bool>>());
  idx_to_children.back().reserve(children.size());

  options.insert_child(parent_idx, idx, self->options(nid, id));

  for(auto const& [down_nid, down_id]: children) {
    bool did_add = insert(idx, down_nid, down_id);
    idx_to_children[idx].emplace_back(down_nid, down_id, did_add);
  }
  return true;
}

uint64_t dyn_solver_t::tree_t::cost_node(
  int idx, int const& rank) const
{
  // For each compute-input in idx,
  //   If the compute-input is part of this tree, then the corresponding
  //   move cost will be included as an edge cost.
  //   Otherwise, include any move cost that might occur.
  uint64_t total = 0;
  for(auto const& [down_nid, down_id, is_in_tree]: idx_to_children[idx]) {
    if(!is_in_tree) {
      if(!self->has_at(down_nid, down_id, rank)) {
        total += self->relations[down_nid].tensor_size();
      }
    }
  }
  return total;
}

uint64_t dyn_solver_t::tree_t::cost_edge(
  int idx_up,   int const& rank_up,
  int idx_down, int const& rank_down) const
{
  if(rank_up == rank_down) {
    return 0;
  }
  auto const& [down_nid, _0] = idx_to_id[idx_down];
  return self->relations[down_nid].tensor_size();
}

void dyn_solver_t::tree_t::solve_and_update() {
  using namespace std::placeholders;

  tree::f_cost_node_t<int> f_cost_node = std::bind(
    &dyn_solver_t::tree_t::cost_node,
    this,
    _1, _2);

  tree::f_cost_edge_t<int> f_cost_edge = std::bind(
    &dyn_solver_t::tree_t::cost_edge,
    this,
    _1, _2, _3, _4);

  tree::tree_t<int> solution = tree::solve(options, f_cost_node, f_cost_edge);

  // Update the solver
  auto const& dag = self->relations[0].dag;
  for(int idx = 0; idx != idx_to_id.size(); ++idx) {
    int const& compute_rank = solution[idx];
    auto const& [nid, id] = idx_to_id[idx];
    node_t const& node = dag[nid];

    // Tell placements the compute ranks
    self->placements[nid].computes[id] = compute_rank;
    self->placements[nid].locs[id].insert(compute_rank);
    // ^ make sure to update the locations too

    // Decrement the load limit, to keep things load balanced
    self->load_limits[nid][compute_rank]--;

    // If this node is a join, the inputs get moved here too.
    // If this node is an agg or reblock, the inputs do not get moved here.
    // This node is not an input.
    if(node.type == node_t::node_type::join) {
      // "Move" the inputs here
      for(auto const& [down_nid, down_id, _uu]: idx_to_children[idx])
      {
        self->placements[down_nid].locs[down_id].insert(compute_rank);
      }
    }
  }
  // NOTE: placements has an invariant that if compute is set, locs has size > 1.
  //       This invariant is not necessarily true during the above for loop.
}

}}

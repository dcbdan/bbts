#include "greedy_placement.h"

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
      // Touches don't actually move the whole, tensor,
      // they move some part of the tensor.
      if(dag[nid].type != node_t::node_type::reblock) {
        placements[nid].locs[idx].insert(rank);
      }
      total += relations[nid].tensor_size();
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

    int num_blocks = relations[nid].get_num_blocks();
    for(int idx = 0; idx != num_blocks; ++idx) {
      if(dag[nid].type == node_t::node_type::agg) {
        move_to_reduce(nid, idx);
      } else {
        int rank = placements[nid].computes[idx];
        auto const& inputs = relations[nid].get_inputs(idx);
        for(auto const& [input_nid, input_idx]: inputs) {
          assure_moved_to(input_nid, input_idx, rank);
        }
      }
    }
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

}}

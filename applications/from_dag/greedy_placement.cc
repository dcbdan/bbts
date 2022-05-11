#include "greedy_placement.h"

namespace bbts { namespace dag {

struct compute_score_t {
  compute_score_t(
    relations_t const& relations):
      relations(relations)
  {}

  uint64_t operator()(
    vector<placement_t> const& placements,
    vector<tuple<nid_t, int>> const& inputs,
    int rank) const
  {
    uint64_t ret = 0;

    for(auto const& [nid, idx]: inputs) {
      auto const& current_locs = placements[nid].locs[idx];
      if(current_locs.count(rank) == 0) {
        ret += relations[nid].tensor_size();
      }
    }

    return ret;
  }

private:
  relations_t const& relations;
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
  vector<placement_t> ret;
  ret.reserve(relations.size());
  for(nid_t nid = 0; nid != relations.size(); ++nid) {
    if(relations[nid].is_no_op()) {
      ret.emplace_back();
    } else {
      ret.emplace_back(relations[nid].get_num_blocks());
    }
  }

  // Round robin assign all inputs
  for(nid_t nid = 0; nid != relations.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.downs.size() == 0) {
      size_t num_blocks = relations[nid].get_num_blocks();
      ret[nid].computes = vector<int>(          num_blocks);
      ret[nid].locs     = vector<std::set<int>>(num_blocks);
      for(int i = 0; i != ret[nid].computes.size(); ++i) {
        ret[nid].locs[i].insert(offset_rank);
        ret[nid].computes[i] = offset_rank;

        increment_offset_rank();
      }
    }
  }

  compute_score_t compute_score(relations);

  // For each non input, compute relation, assing the compute locations
  // and update where each block ends up being moved to
  for(nid_t const& nid: dag.breadth_dag_order()) {
    // Skip no ops and input nodes
    if(relations[nid].is_no_op() || dag[nid].downs.size() == 0) {
      continue;
    }

    int num_blocks = relations[nid].get_num_blocks();
    int min_per_rank = num_blocks / num_nodes;

    vector<int> counts;
    if(dag[nid].downs.size() == 1 && dag[nid].type == node_t::node_type::join) {
      // If this is a join with one input, each block should end up happening in the same
      // place as the input block.
      vector<int> dummy_bid(dag[nid].dims.size(), 0);
      auto [input_nid, _0] = relations[nid].get_inputs(dummy_bid)[0];
      counts = vector<int>(num_nodes, 0);
      for(int const& rank: ret[input_nid].computes) {
        counts[rank]++;
      }
    } else {
      // Otherwise, use the offset rank to determine how many blocks
      // will be done by each rank
      counts = vector<int>(num_nodes, min_per_rank);
      int rem = num_blocks % num_nodes;
      for(int i = 0; i != rem; ++i) {
        counts[offset_rank]++;
        increment_offset_rank();
      }
    }

    indexer_t indexer(relations[nid].partition);
    do {
      auto const& bid = indexer.idx;
      auto const& idx = relations[nid].bid_to_idx(bid);

      auto inputs = relations[nid].get_inputs(bid);
      int best_rank = -1;
      uint64_t best_score;
      for(int rank = 0; rank != counts.size(); ++rank) {
        if(counts[rank] > 0) {
          if(best_rank < 0) {
            best_rank = rank;
            best_score = compute_score(ret, inputs, rank);
          } else {
            uint64_t score = compute_score(ret, inputs, rank);
            if(score < best_score) {
              best_rank = rank;
              best_score = score;
            }
          }
        }
      }

      // The compute location has been chosen, update the information

      // set compute
      ret[nid].computes[idx] = best_rank;
      ret[nid].locs[idx].insert(best_rank);

      // update counts
      counts[best_rank]--;

      // update the input ranks for tensors that got moved..
      // reblock and agg don't count because
      // touch   and reduce don't move the input tensors
      if(dag[nid].type == node_t::node_type::join) {
        for(auto const& [input_nid, input_idx]: inputs) {
          ret[input_nid].locs[input_idx].insert(best_rank);
        }
      }
    } while (indexer.increment());
  }

  return ret;
}

}}

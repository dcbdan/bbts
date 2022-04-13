#include "generate.h"
#include "misc.h"
#include "../../src/utils/expand_indexer.h"

#include <algorithm>
#include <numeric>

#define DCB01(x) // std::cout << x << std::endl;

namespace bbts { namespace dag {

using utils::expand::expand_indexer_t;
using utils::expand::column_major_expand_t;

int select_node_t::select_score(vector<int> const& scores) {
  auto m = *std::max_element(scores.begin(), scores.end());

  vector<int> which;
  which.reserve(scores.size());
  for(int i = 0; i != scores.size(); ++i) {
    if(scores[i] == m) {
      which.push_back(i);
    }
  }

  return select_from(which);
}

int select_node_t::select_from(vector<int> const& select_from_)
{
  int which = select_from_.front();
  uint64_t score = counts[which];

  for(int i = 1; i < select_from_.size(); ++i) {
    int const& which_maybe = select_from_[i];
    uint64_t const& score_maybe = counts[which_maybe];

    if(score_maybe < score) {
      score = score_maybe;
      which = which_maybe;
    }
  }

  // now increment
  counts[which]++;

  return which;
}

generate_commands_t::generate_commands_t(
  dag_t const& dag,
  vector<partition_info_t> const& info,
  std::function<ud_impl_id_t(int)> get_ud,
  int num_nodes):
    dag(dag),
    info(info),
    get_ud(get_ud),
    num_nodes(num_nodes),
    selector(num_nodes),
    _command_id(0),
    _tid(0),
    _priority(std::numeric_limits<int32_t>::max())
{
  relations.reserve(dag.size());
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    relations.emplace_back(this, nid);
  }

  vector<nid_t> idxs = priority_dag_order();

  // recursing at the ouptut nodes, add the priorities
  for(nid_t which: idxs) {
    if(dag[which].ups.size() == 0) {
      add_priority(which);
    }
  }

  for(nid_t which: idxs) {
    add_node(which);
  }
  //// Add the input nodes everywhere
  //for(nid_t which: idxs) {
  //  if(dag[which].downs.size() == 0) {
  //    add_input_node_everywhere(which);
  //  }
  //}

  //// For the rest of the nodes, add the items in the relations
  //// wherever they end up
  //for(nid_t which: idxs) {
  //  if(dag[which].downs.size() > 0) {
  //    add_node(which);
  //  }
  //}
}

bool priority_was_set(int32_t p) {
  return p != std::numeric_limits<int32_t>::min();
}

void generate_commands_t::add_input_node_everywhere(nid_t nid) {
  node_t const& node = dag[nid];
  assert(node.type == node_t::node_type::input);

  vector<loc_t> all_other_locs(num_nodes-1);
  std::iota(all_other_locs.begin(), all_other_locs.end(), 1);

  // for each bid, add the command and get the output
  indexer_t indexer(relations[nid].partition);
  do {
    auto const& bid = indexer.idx;

    tid_t tid = next_tid();
    for(int which_node = 0; which_node != num_nodes; ++which_node) {
      tid_loc_t cur{ tid, which_node };

      auto params = node.get_bbts_params();
      // input kernels in libbarb_cutensor require
      // appending for each rank
      //   (which blk, local dim size, output dim size)
      // as parameters
      using uint_type = decltype(bbts::command_param_t().u);
      auto const& ps = relations[nid].partition;
      for(int r = 0; r != bid.size(); ++r) {
        params.push_back({ .u = static_cast<uint_type>(bid[r])               });
        params.push_back({ .u = static_cast<uint_type>(node.dims[r] / ps[r]) });
        params.push_back({ .u = static_cast<uint_type>(node.dims[r])         });
      }

      input_commands.emplace_back(command_t::create_apply(
        next_command_id(),
        get_ud(node.kernel),
        false,
        params,
        {},
        {cur}));
      input_commands.back()->priority = relations[nid].priority(bid);
    }

    relations[nid][bid] = {tid, 0};
    if(num_nodes > 1) {
      // they aren't really being moved, but that is ok
      moved_to_locs[tid] = all_other_locs;
    }
  } while (indexer.increment());
}

void generate_commands_t::add_node(nid_t nid) {
  // don't do anything if this is actually a no op!
  // get_inputs will reach past no ops.
  if(relations[nid].is_no_op) {
    return;
  }

  node_t const& node = dag[nid];

  // Here are some things specific to reblocking, but don't need to be
  // cereated over and over for every block
  std::unique_ptr<expand_indexer_t> expand_indexer_ptr(nullptr);
  if(node.type == node_t::node_type::reblock) {
    auto const& partition_here = relations[nid].partition;
    auto const& partition_down = relations[node.downs[0]].partition;

    expand_indexer_ptr = std::unique_ptr<expand_indexer_t>(
      new expand_indexer_t(
        partition_here,
        partition_down));

  }

  // for each bid, add the command and get the output
  indexer_t indexer(relations[nid].partition);
  do {
    auto const& bid = indexer.idx;

    int priority = 0; // relations[nid].priority(bid);
    assert(priority_was_set(priority));

    vector<tid_loc_t> inputs = relations[nid].get_inputs(bid);

    // TODO: This location-choosing strategy is being used because
    // it is the easiest solution. When it is verified that this
    // could be better, and determined how it could be better,
    // change it.
    vector<int> num_from_node(num_nodes, 0);
    // ^ for each location, count the number of tensors that
    //   already exist at that location
    for(auto const& [tid, loc]: inputs) {
      num_from_node[loc]++;
      for(auto const& moved_to_loc: moved_to_locs[tid]) {
        num_from_node[moved_to_loc]++;
      }
    }
    // pick one of the locations that has the most number of tensors
    // already available
    int compute_location = selector.select_score(num_from_node);

    // Now we have all the inputs, figure out what the node is and use that
    // to create the command and get a new tid.

    if(node.type == node_t::node_type::input)
    {
      tid_loc_t cur{ next_tid(), compute_location };

      auto params = node.get_bbts_params();
      // input kernels in libbarb_cutensor require
      // appending for each rank
      //   (which blk, local dim size, output dim size)
      // as parameters
      using uint_type = decltype(bbts::command_param_t().u);
      auto const& ps = relations[nid].partition;
      for(int r = 0; r != bid.size(); ++r) {
        params.push_back({ .u = static_cast<uint_type>(bid[r])               });
        params.push_back({ .u = static_cast<uint_type>(node.dims[r] / ps[r]) });
        params.push_back({ .u = static_cast<uint_type>(node.dims[r])         });
      }

      input_commands.emplace_back(command_t::create_apply(
        next_command_id(),
        get_ud(node.kernel),
        false,
        params,
        {},
        {cur}));
      input_commands.back()->priority = priority;

      relations[nid][bid] = cur;
    }
    else if(node.type == node_t::node_type::join)
    {
      // for each input, move it to the location if necessary
      vector<tid_loc_t> apply_inputs;
      apply_inputs.reserve(inputs.size());
      for(auto const& [tid,loc]: inputs) {
        if(loc != compute_location) {
          // move it if it hasn't yet been moved
          assure_moved_to(commands, tid, loc, compute_location);
        }
        apply_inputs.push_back({tid, compute_location});
      }

      tid_loc_t cur{ next_tid(), compute_location };

      commands.emplace_back(command_t::create_apply(
        next_command_id(),
        get_ud(node.kernel),
        false,
        node.get_bbts_params(),
        apply_inputs,
        {cur}));
      commands.back()->priority = priority;

      relations[nid][bid] = cur;
    }
    // Create a reduce command
    else if(node.type == node_t::node_type::agg)
    {
      tid_loc_t cur{ next_tid(), compute_location };

      commands.emplace_back(command_t::create_reduce(
        next_command_id(),
        get_ud(node.kernel),
        false,
        node.get_bbts_params(),
        inputs,
        cur));
      commands.back()->priority = priority;

      relations[nid][bid] = cur;
    }
    else if(node.type == node_t::node_type::reblock)
    {
      expand_indexer_t const& expand_indexer = *expand_indexer_ptr;

      // the same reblock params are used for each of the inputs
      vector<command_param_t> reblock_params;
      reblock_params.reserve(4*node.dims.size());
      auto insert = [&](int x) {
        reblock_params.push_back({ .u = static_cast<uint32_t>(x) });
      };
      auto const& partition_here = relations[nid].partition;
      auto const& partition_down = relations[node.downs[0]].partition;
      for(int r = 0; r != node.dims.size(); ++r) {
        insert(node.dims[r]);
        insert(partition_down[r]);
        insert(partition_here[r]);
        insert(bid[r]);
      }

      tid_loc_t cur{ next_tid(), compute_location };

      for(int which_input = 0; which_input != inputs.size(); ++which_input) {
        auto [tid,loc] = inputs[which_input];

        if(was_moved_to(tid, compute_location)) {
          loc = compute_location;
        }

        column_major_expand_t expand(expand_indexer.get_expand_dim(
          node.dims,
          expand_indexer.get_which_input(bid, which_input),
          bid));

        bool compact_is_new = false;
        tid_t compact_tid = tid;
        if(!expand.is_compact_inn()) {
          compact_tid = next_tid();
          compact_is_new = true;

          // do the compact into compact_tid
          commands.emplace_back(command_t::create_compact(
            next_command_id(),
            get_ud(node.kernel),
            false,
            which_input,
            reblock_params,
            {tid, loc},
            {compact_tid, loc}));
          commands.back()->priority = priority;
        }
        // if the compact input needs to be moved, move it
        if(loc != compute_location) {
          assure_moved_to(commands, compact_tid, loc, compute_location);
          if(compact_is_new) {
            commands.emplace_back(command_t::create_delete(
              next_command_id(),
              {tid_loc_t{compact_tid, loc}}));
          }
        }
        // now a compact item exists at the destination location,
        // so do the touch
        commands.emplace_back(command_t::create_touch(
          next_command_id(),
          get_ud(node.kernel),
          false,
          which_input,
          inputs.size(),
          reblock_params,
          {tid_loc_t{compact_tid, compute_location}},
          cur));
        commands.back()->priority = priority;
        if(compact_is_new) {
          commands.emplace_back(command_t::create_delete(
            next_command_id(),
            {tid_loc_t{compact_tid, compute_location}}));
        }
      }

      relations[nid][bid] = cur;
    } else {
      throw std::runtime_error("should not reach");
    }

  } while (indexer.increment());
}

void generate_commands_t::add_priority(nid_t nid) {
  // For each block, recursively add priorities
  indexer_t indexer(relations[nid].partition);
  do {
    auto const& bid = indexer.idx;
    relations[nid].add_priority(bid);
  } while (indexer.increment());
}

tuple<vector<command_ptr_t>, vector<command_ptr_t>>
generate_commands_t::extract()
{
  tuple<vector<command_ptr_t>, vector<command_ptr_t>> ret{
    std::move(input_commands),
    std::move(commands)};
  input_commands.resize(0);
  commands.resize(0);
  return std::move(ret);
}

generate_commands_t::relation_t::relation_t(generate_commands_t* self_, nid_t nid_):
  self(self_),
  nid(nid_),
  partition(self_->info[nid_].blocking),
  is_no_op(_is_no_op())
{
  int n = is_no_op ? 0 : product(partition);
  tid_locs = vector<tid_loc_t>(n, {-1,-1});
  priorities = vector<int32_t>(n, std::numeric_limits<int32_t>::min());
}

vector<tuple<nid_t, vector<int>>>
generate_commands_t::relation_t::get_bid_inputs(vector<int> const& bid) const
{
  assert(bid.size() == partition.size());

  node_t const& node = self->dag[nid];

  if(node.type == node_t::node_type::agg) {
    vector<tuple<nid_t, vector<int>>> ret;

    // basically, cartesian product over the agg dimensions.
    nid_t join_nid = node.downs[0];
    node_t const& join_node = self->dag[join_nid];
    auto const& aggs = join_node.aggs;
    auto const& join_blocking = self->info[join_nid].blocking;

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

    if(is_no_op) {
      return {{input_nid, bid}};
    } else {
      vector<tuple<nid_t, vector<int>>> ret;
      // the expander can figure out which inputs touch
      // this output.
      expand_indexer_t e_indexer(
        // this is the input partitioning
        self->relations[input_nid].partition,
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

  if(node.type == node_t::node_type::join) {
    vector<tuple<nid_t, vector<int>>> ret;
    ret.reserve(node.downs.size());

    for(auto const& input_nid: node.downs) {
      vector<int> input_bid = self->dag.get_out_for_input(bid, nid, input_nid);
      ret.emplace_back(input_nid, input_bid);
    }

    return ret;
  }

  if(node.type == node_t::node_type::input) {
    return {};
  }

  assert(false);
  return {};
}

// Go into the input relations and get the input tid_locs
vector<tid_loc_t>
generate_commands_t::relation_t::get_inputs(vector<int> const& bid)
{
  assert(bid.size() == partition.size());

  vector<tid_loc_t> ret;

  node_t const& node = self->dag[nid];

  if(node.type == node_t::node_type::agg) {
    // basically, cartesian product over the agg dimensions.
    nid_t join_nid = node.downs[0];
    node_t const& join_node = self->dag[join_nid];
    auto const& aggs = join_node.aggs;
    auto const& join_blocking = self->info[join_nid].blocking;

    vector<int> agg_szs;
    agg_szs.reserve(aggs.size());
    for(int which: aggs) {
      agg_szs.push_back(join_blocking[which]);
    }

    indexer_t indexer(agg_szs);

    ret.reserve(product(agg_szs));
    do {
      auto inc_bid = dag_t::combine_out_agg(aggs, bid, indexer.idx);
      ret.push_back(self->relations[join_nid][inc_bid]);
    } while(indexer.increment());
  }

  if(node.type == node_t::node_type::reblock) {
    nid_t input_nid = node.downs[0];

    if(is_no_op) {
      ret = {self->relations[input_nid][bid]};
    } else {
      // the expander can figure out which inputs touch
      // this output.
      expand_indexer_t e_indexer(
        // this is the input partitioning
        self->relations[input_nid].partition,
        // this is the output partitioning
        partition);

      // get all those inputs
      auto inputs = expand_indexer_t::cartesian(e_indexer.get_inputs(bid));

      ret.reserve(inputs.size());
      for(auto const& input_bid: inputs) {
        ret.push_back(self->relations[input_nid][input_bid]);
      }
    }
  }

  if(node.type == node_t::node_type::join) {
    ret.reserve(node.downs.size());

    for(auto const& input_nid: node.downs) {
      vector<int> input_bid = self->dag.get_out_for_input(bid, nid, input_nid);
      ret.push_back(self->relations[input_nid][input_bid]);
    }
  }

  if(node.type == node_t::node_type::input) {
    ret = {};
  }

  // Uninitialized tids are negative.. Make sure
  // no uninitialized tids were retrieved.
  // Same with locations.
  for(auto const& [t,l]: ret) {
    assert(t >= 0);
    assert(l >= 0);
  }

  return ret;
}

// If this node is a no op, then reach past this relation.
tid_loc_t& generate_commands_t::relation_t::operator[](vector<int> const& bid) {
  assert(bid.size() == partition.size());
  if(is_no_op) {
    node_t const& node = self->dag[nid];

    if(node.type == node_t::node_type::reblock) {
      // if this is a no op, the input relation has the same shape,
      // so use the same bid
      nid_t input_nid = node.downs[0];
      return self->relations[input_nid][bid];
    }

    if(node.type == node_t::node_type::agg) {
      // if this is a no op, then the agg dims are all 1,
      // so the inc_bid needs to be zero there.. so if j is agg,
      // ijk->ik, then bid = (1,3) -> inc_bid = (1,0,3)
      nid_t join_nid = node.downs[0];
      node_t const& join_node = self->dag[join_nid];

      // just in case
      assert(join_node.type == node_t::node_type::join);

      auto const& aggs = join_node.aggs;

      vector<int> inc_bid = dag_t::combine_out_agg(
        aggs,
        bid,
        vector<int>(aggs.size(), 0));

      return self->relations[join_nid][inc_bid];
    }

    throw std::runtime_error("should not reach");
  }

  // otherwise, we find the tid, column major ordering and all that
  return tid_locs[bid_to_idx(bid)];
}

// column major ordering
int generate_commands_t::relation_t::bid_to_idx(vector<int> const& bid) const {
  vector<int> const& parts = self->info[nid].blocking;
  int idx = 0;
  int p = 1;
  for(int r = 0; r != bid.size(); ++r) {
    idx += p * bid[r];
    p   *= parts[r];
  }
  return idx;
}

int32_t generate_commands_t::relation_t::priority(vector<int> const& bid) const {
  assert(!is_no_op);
  return priorities[bid_to_idx(bid)];
}

void generate_commands_t::relation_t::print(std::ostream& os) const {
  std::stringstream s;
  s << (self->dag[nid]);
  auto str = s.str();

  std::string spaces =
    60 > str.size()                   ?
    std::string(60 - str.size(), ' ') :
    std::string(10, ' ')              ;

  os << str << spaces << "  &  blocking: ";
  print_list(os, partition);
}

bool generate_commands_t::relation_t::_is_no_op() {
  node_t const& node = self->dag[nid];
  if(node.type == node_t::node_type::input) {
    return false;
  }
  if(node.type == node_t::node_type::join) {
    return false;
  }
  if(node.type == node_t::node_type::reblock) {
    // if these are the same, there is no reblock
    auto const& my_blocking    = self->info[nid].blocking;
    auto const& input_blocking = self->info[node.downs[0]].blocking;
    return my_blocking == input_blocking;
  }
  if(node.type == node_t::node_type::agg) {
    auto const& my_blocking    = self->info[nid].blocking;
    auto const& join_blocking = self->info[node.downs[0]].blocking;
    // if there is nothing to agg, this is a no op
    return product(my_blocking) == product(join_blocking);
  }
  throw std::runtime_error("should not reach");
  return true;
}

void generate_commands_t::relation_t::add_priority(vector<int> const& bid)
{
  // Was this already set? then don't recurse and exit.
  // Otherwise set the priority if this is an op and recurse.

  if(!is_no_op) {
    auto& p = priorities[bid_to_idx(bid)];
    if(priority_was_set(p)) {
      return;
    } else {
      p = self->next_priority();
    }
  }
  for(auto const& [child_nid, input_bid]: get_bid_inputs(bid)) {
    self->relations[child_nid].add_priority(input_bid);
  }
}

bool generate_commands_t::was_moved_to(tid_t tid, loc_t loc) {
  for(auto const& other_loc: moved_to_locs[tid]) {
    if(other_loc == loc) {
      return true;
    }
  }
  return false;
}

void generate_commands_t::assure_moved_to(
  vector<command_ptr_t>& cmds,
  tid_t tid, loc_t from, loc_t to)
{
  if(was_moved_to(tid, to)) {
    return;
  }
  cmds.emplace_back(command_t::create_move(
    next_command_id(),
    {tid, from},
    {tid, to}));
  moved_to_locs[tid].push_back(to);
}

}}




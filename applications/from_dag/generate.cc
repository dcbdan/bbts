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
    _tid(0)
{
  relations.reserve(dag.size());
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    relations.emplace_back(this, nid);
  }

  vector<nid_t> idxs = priority_dag_order();

  // Starting with nodes of the lowest priorty, add them to
  // the relation dag
  for(nid_t which: idxs) {
    add_node(which);
  }
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

    vector<tid_loc_t> inputs = relations[nid].get_inputs(bid);

    // TODO: This location-choosing strategy is being used because
    // it is the easiest solution. When it is verified that this
    // could be better, and determined how it could be better,
    // change it.
    vector<int> num_from_node(num_nodes, 0);
    for(auto const& [_1, loc]: inputs) {
      num_from_node[loc]++;
    }
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

      relations[nid][bid] = cur;
    }
    else if(node.type == node_t::node_type::join)
    {
      // for each input, move it to the location if necessary
      vector<tid_loc_t> apply_inputs;
      apply_inputs.reserve(inputs.size());
      for(auto const& [tid,loc]: inputs) {
        if(loc == compute_location) {
          apply_inputs.push_back({tid, compute_location});
        } else {
          commands.emplace_back(command_t::create_move(
            next_command_id(),
            {tid, loc},
            {tid, compute_location}));
          apply_inputs.push_back({tid, compute_location});
        }
      }

      tid_loc_t cur{ next_tid(), compute_location };

      commands.emplace_back(command_t::create_apply(
        next_command_id(),
        get_ud(node.kernel),
        false,
        node.get_bbts_params(),
        apply_inputs,
        {cur}));
      (commands.back())->priority = info[nid].start * -1;

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
      (commands.back())->priority = info[nid].start * -1;

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
        auto const& [tid,loc] = inputs[which_input];

        column_major_expand_t expand(expand_indexer.get_expand_dim(
          node.dims,
          expand_indexer.get_which_input(bid, which_input),
          bid));

        ///////////////////////////////////////////////////////////////////////

        //bool cleanup_touching_tid = false;
        //tid_t touching_tid = tid;
        //if(loc != compute_location) {
        //  if(expand.is_compact_inn()) {
        //    // if it is already compact, just send it
        //    commands.emplace_back(command_t::create_move(
        //      next_command_id(),
        //      {tid, loc},
        //      {tid, compute_location}));
        //  } else {
        //    // if it can be compacted, compact it
        //    tid_t compact_tid = next_tid();
        //    touching_tid = compact_tid;

        //    // do the compact into compact_tid
        //    commands.emplace_back(command_t::create_compact(
        //      next_command_id(),
        //      get_ud(node.kernel),
        //      false,
        //      which_input,
        //      reblock_params,
        //      {tid, loc},
        //      {compact_tid, loc}));

        //    // move the compacted tensor to the compute location
        //    commands.emplace_back(command_t::create_move(
        //      next_command_id(),
        //      {compact_tid, loc},
        //      {compact_tid, compute_location}));

        //    // now delete the compacted tensor
        //    cleanup_touching_tid = true;
        //    commands.emplace_back(command_t::create_delete(
        //      next_command_id(),
        //      {tid_loc_t{compact_tid, loc}}));
        //  }
        //}

        //// do the touch
        //commands.emplace_back(command_t::create_touch(
        //  next_command_id(),
        //  get_ud(node.kernel),
        //  false,
        //  which_input,
        //  inputs.size(),
        //  reblock_params,
        //  {tid_loc_t{touching_tid, compute_location}},
        //  cur));
        //(commands.back())->priority = info[nid].start * -1;

        //// if we eneded up moving this guy, delete it
        //if(cleanup_touching_tid) {
        //  commands.emplace_back(command_t::create_delete(
        //    next_command_id(),
        //    {tid_loc_t{touching_tid, compute_location}}));
        //}

        ///////////////////////////////////////////////////////////////////////

        // The expand kernel has two parts:
        //   compact and uncompact.
        // Compact compresses the input into a contiguous region.
        // Uncompact takes that contiguous region and writes to the output.
        //
        // If the input is already contiguous, no compact is needed to happen
        // and the write can happen directly to the output.
        //
        // But if the expand kernel is called and a compact does need to happen,
        // it has to allocate memory to hold the compact value.
        //
        // Here, we split the compact and uncompact into separate kernel calls so
        // that the tos can have its memory. And we always delete the
        // compact temporaries with the tos too.

        // If the input needs to be compacted, compact it.
        // Anywhere a new compact tensor created, make sure it
        // gets deleted.
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
          (commands.back())->priority = info[nid].start * -1;
        }
        // if the compact input needs to be moved, move it
        if(loc != compute_location) {
          commands.emplace_back(command_t::create_move(
            next_command_id(),
            {compact_tid, loc},
            {compact_tid, compute_location}));
          (commands.back())->priority = info[nid].start * -1;
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
        (commands.back())->priority = info[nid].start * -1;
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
  tid_locs = vector<tid_loc_t>(
    is_no_op ? 0 : product(partition),
    {-1, -1});
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

    for(auto const& reblock_nid: node.downs) {
      vector<int> reblock_bid = self->dag.get_reblock_out(bid, reblock_nid);
      ret.push_back(self->relations[reblock_nid][reblock_bid]);
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
  // TODO: put this in a function somewhere...
  vector<int> const& parts = self->info[nid].blocking;
  int idx = 0;
  int p = 1;
  for(int r = 0; r != bid.size(); ++r) {
    idx += p * bid[r];
    p   *= parts[r];
  }
  return tid_locs[idx];
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

}}




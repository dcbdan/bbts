#include "placement.h"

#define DCB01(x) // std::cout << __LINE__ << " " << x << std::endl;
#define DCB02(x) // std::cout << __LINE__ << " " << x << std::endl;
#define DCB03(x) // std::cout << __LINE__ << " " << x << std::endl;

namespace bbts { namespace dag {

using namespace Gecode;

Placement::Placement(
  relations_t const& relations_,
  vector<placement_t> const& info_,
  int num_nodes_,
  int cover_size_):
    IntMinimizeSpace(),
    relations(relations_),
    info(info_),
    num_nodes(num_nodes_),
    cover_size(cover_size_),
    rnd(0)
{
  DCB01("A with num nodes " << num_nodes);
  _is_completely_covered = _set_covering();

  DCB01("B");
  vars = vector<relvar_ptr>(relations.size());
  for(auto& ptr: vars) {
    ptr = nullptr;
  }

  for(nid_t const& nid: covering()) {
    vars[nid] = std::unique_ptr<relvar_t>(new relvar_t(this, nid));
  }
  DCB01("C");

  for(nid_t const& nid: covering()) {
    vars[nid]->set_constraints();
  }

  DCB01("D");
  IntVarArgs costs;
  for(nid_t const& nid: covering()) {
    relvar_t& relvar = *vars[nid];
    for(int i = 0; i != relvar.size(); ++i) {
      costs << relvar.costs[i];
    }
  }
  total_move = expr(*this, sum(costs));

  DCB01("E");

  _set_branching();
  DCB01("F");
}

Placement::Placement(Placement& other):
  IntMinimizeSpace(other),
  relations(other.relations),
  info(other.info),
  num_nodes(other.num_nodes),
  cover_size(other.cover_size),
  rnd(other.rnd),
  _covering(other._covering),
  _is_completely_covered(other._is_completely_covered)
{
  total_move.update(*this, other.total_move);

  vars = vector<relvar_ptr>(info.size());
  for(auto& ptr: vars) {
    ptr = nullptr;
  }

  for(nid_t const& nid: covering()) {
    vars[nid] = relvar_ptr(new relvar_t(this, *other.vars[nid]));
  }
}

Placement::relvar_t::relvar_t(Placement* self_, nid_t nid_):
  self(self_), nid(nid_)
{
  DCB01("relvar size " << size());
  int num_nodes = self->num_nodes;

  // each var takes on a value from 0 to num_nodes-1.
  if(!fixed_computes()) {
    computes = IntVarArray(*self, size(),
      0, num_nodes-1);
  }

  costs = IntVarArray(*self, size(), 0, 1000000);
}

Placement::relvar_t::relvar_t(Placement* self_, relvar_t& other):
  self(self_), nid(other.nid)
{
  if(!fixed_computes()) {
    computes.update(*self, other.computes);
  }
  costs.update(*self, other.costs);
}

// Make sure that there is a minimum number of nodes
// being computed at each location
void Placement::relvar_t::load_balance() {
  DCB01("load balance enter");
  int n = size();
  int num_nodes = self->num_nodes;

  int min_per_node = n / num_nodes;

  DCB02("min_per_node: " << min_per_node);

  for(int rank = 0; rank != num_nodes; ++rank) {
    BoolVarArgs args;
    for(int i = 0; i != n; ++i) {
      args << expr(*self, computes[i] == rank);
    }
    rel(*self, sum(args) >= min_per_node);
  }
  DCB01("load balance exit");
}

void Placement::set_cost_t::operator()(Space& home) const {
  Placement& self = static_cast<Placement&>(home);
  Placement::relvar_t& r = static_cast<Placement::relvar_t&>(*self.vars[nid]);

  std::set<int> all_locs;

  if(r.fixed_computes()) {
    all_locs.insert(self.info[nid].computes[idx]);
  } else {
    all_locs.insert(r.computes[idx].val());
  }

  auto const& outputs = self.relations[nid].get_outputs(idx);
  for(auto const& [output_nid, output_idx]: outputs) {
    if(self.vars[output_nid]) {
      all_locs.insert(self.vars[output_nid]->computes[output_idx].val());
    }
  }

  int cost = all_locs.size() - 1; // TODO: use tensor size

  Int::IntView _cost = r.costs[idx];

  ModEvent _ev = _cost.eq(self, cost);

  if(_ev == Int::ME_INT_FAILED)
  {
    self.fail();
  }
}

// For each block, when it is determined where this guy ends up
// getting moved, set the costs
void Placement::relvar_t::set_costs() {
  indexer_t indexer(self->relations[nid].partition);
  do {
    auto const& bid = indexer.idx;
    int idx = self->relations[nid].bid_to_idx(bid);

    vector<tuple<nid_t, int>> const& outputs = self->relations[nid].get_outputs(idx);
    IntVarArgs cs;

    if(!fixed_computes()) {
      cs << computes[idx];
    }

    for(auto const& [output_nid, output_idx]: outputs) {
      if(self->vars[output_nid]) {
        cs << self->vars[output_nid]->computes[output_idx];
      }
    }

    // Once cs are set, set costs[idx].
    wait(*self, cs, set_cost_t(nid, idx));

  } while (indexer.increment());
}

vector<nid_t> Placement::get_inputs(nid_t nid) const {
  auto const& dag = relations[0].dag;

  vector<nid_t> ret;
  for(nid_t down: dag[nid].downs) {
    if(relations[down].is_no_op()) {
      auto is = get_inputs(down);
      for(nid_t const& i: is) {
        ret.push_back(i);
      }
    } else {
      ret.push_back(down);
    }
  }

  return ret;
}

void Placement::_add(nid_t nid, std::set<nid_t>& so_far, int& num_added) const {
  if(relations[nid].is_no_op()) {
    throw std::runtime_error("should not happen!");
  }

  auto [_0, did_insert] = so_far.insert(nid);
  if(did_insert && !info[nid].computes_set()) {
    num_added++;
  } else {
    // If (1) nothing was inserted or
    //    (2) this node already has the computes set, don't recurse.
    return;
  }

  for(nid_t down: get_inputs(nid)) {
    _add(down, so_far, num_added);
  }
}

// For every node that is not fixed or set and is an op, in order,
//   add until you get to a fixed node.
bool Placement::_set_covering() {
  DCB01("SC A");

  auto const& dag = relations[0].dag;
  int num_added = 0;
  std::set<nid_t> so_far;
  bool ret = true;
  DCB01("SC B");
  for(nid_t const& nid: dag.breadth_dag_order()) {
    DCB01("SC." << nid);
    if(!relations[nid].is_no_op() && !info[nid].computes_set()) {
      _add(nid, so_far, num_added);
      if(num_added > cover_size) {
        ret = false;
        break;
      }
    }
  }

  DCB01("SC C");

  _covering.reserve(so_far.size());
  for(nid_t const& nid: so_far) {
    _covering.push_back(nid);
  }

  DCB01("SC D");

  return ret;
}

void Placement::_set_branching() {
  DCB01("set branching enter");
  for(nid_t const& nid: covering()) {
    branch(*this, vars[nid]->computes, INT_VAR_RND(rnd), INT_VAL_RND(rnd));
  }
  DCB01("set branching exit");
}

// NOTE: Search::Options is not a proper data type, it contains pointers to stuff
//       that get modified and deleted. So call this function to get a "fresh"
//       object and don't use search options more than once.
Search::Options build_search_options() {
  int num_threads = 24;
  int search_time_per_cover = 30000;
  int search_restart_scale = Search::Config::slice;

  Search::Options so;

  // Basically copying runMeta in driver/Script.hpp for the
  // solution case.
  //
  // The interaction between the options is not completely clear.

  so.threads = num_threads;

  so.c_d = Search::Config::c_d;
  so.a_d = Search::Config::a_d;
  so.d_l = Search::Config::d_l;

  // stop on only interrupt or after the time limit
  so.stop = Driver::CombinedStop::create(
                0u , 0u, search_time_per_cover, false);

  so.cutoff  = Search::Cutoff::luby(search_restart_scale);

  so.clone   = false;

  // When this isn't needed, gecode calls installCtrlHandler(false)...
  //Driver::CombinedStop::installCtrlHandler(true);

  return so;
}

Placement* _run(Placement* init) {
  RBS<Placement, BAB> engine(init, build_search_options());

  Placement* ret = nullptr;

  do {
    Placement* ex = engine.next();
    if(!ex) {
      break;
    }
    if(ret) {
      delete ret;
    }
    ret = ex;
    ret->print(std::cout);
  } while (true);

  return ret;
}

vector<placement_t> solve_placement(
  relations_t const& relations,
  int num_nodes,
  int num_cover)
{
  DCB01("SOLVE A");
  vector<placement_t> ret(relations.size());

  // For each input node, give it a round robin assignment;
  // make sure that locs includes the computed at location
  {
    auto const& dag = relations[0].dag;
    int which_rank = 0;
    for(nid_t nid = 0; nid != relations.size(); ++nid) {
      node_t const& node = dag[nid];
      if(node.downs.size() == 0) {
        size_t num_blocks = relations[nid].get_num_blocks();
        ret[nid].computes = vector<int>(          num_blocks);
        ret[nid].locs     = vector<std::set<int>>(num_blocks);
        for(int i = 0; i != ret[nid].computes.size(); ++i) {
          ret[nid].locs[i].insert(which_rank);
          ret[nid].computes[i] = which_rank;

          which_rank++;
          if(which_rank == num_nodes) {
            which_rank = 0;
          }

        }
      }
    }
  }

  Placement* placement = _run(new Placement(relations, ret, num_nodes, num_cover));

  DCB01("SOLVE B");
  if(!placement) {
    throw std::runtime_error("Placement: Could not find a solution!");
  } else {
    for(nid_t const& nid: placement->covering()) {
      if(!ret[nid].computes_set()) {
        ret[nid].computes = placement->get_computes(nid);
      }
      //ret[nid].locs = placement->get_locs(nid);
    }
  }

  DCB01("SOLVE C");
  while(!placement->is_completely_covered()) {
    throw std::runtime_error("rewrite covering and make sure to set locs");

    std::cout << "..." << std::endl;
    Placement* tmp = new Placement(relations, ret, num_nodes, num_cover);

    delete placement;

    placement = _run(tmp);

    if(!placement) {
      throw std::runtime_error("Placement: Could not find a solution!");
    } else {
      for(nid_t const& nid: placement->covering()) {
        if(!ret[nid].computes_set()) {
          ret[nid].computes = placement->get_computes(nid);
        }
        ret[nid].locs = placement->get_locs(nid);
      }
    }
  }
  DCB01("SOLVE D");

  delete placement;

  return ret;
}

vector<int> Placement::get_computes(nid_t nid) {
  DCB03("get_computes");
  vector<int> ret;
  relvar_t& relvar = *vars[nid];
  for(int i = 0; i != relvar.size(); ++i) {
    DCB03("relvar.computes[i].size(): " << relvar.computes[i].size());
    ret.push_back(relvar.computes[i].val());
  }
  return ret;
}

}}

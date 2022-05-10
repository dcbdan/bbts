#pragma once

#include "dag.h"
#include "relation.h"
#include "misc.h"

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/set.hh>

#include <vector>
#include <tuple>
#include <functional>
#include <set>

namespace bbts { namespace dag {

using std::vector;
using std::tuple;
using std::function;

struct placement_t {
  bool computes_set() const {
    return computes.size() > 0;
  }

  // If computes.size() == 0, then
  //   this node is a no op or there is nothing set
  vector<int> computes;
  // If computes.size() != 0 and locs.size() == 0, then
  //   the locs still need to be figured out
  vector<std::set<int>> locs;
};

struct Placement : public Gecode::IntMinimizeSpace {
  Placement(
    vector<relation_t> const& relations_,
    vector<placement_t> const& info_,
    int num_nodes_,
    int num_covered_);

  Placement(Placement& other);

  int const num_nodes;
  int const cover_size;

  vector<nid_t> const& covering() const { return _covering; }

  bool is_completely_covered() const { return _is_completely_covered; }

  virtual void print(std::ostream& os) const {
    os << "Placement cost: " << total_move.min() << std::endl;
  }

  vector<int> get_computes(nid_t nid);
  vector<std::set<int>> get_locs(nid_t nid);

private:
  struct relvar_t {
    relvar_t(Placement* self, nid_t nid);
    relvar_t(Placement* self, relvar_t& other);

    Gecode::IntVarArray computes;
    Gecode::SetVarArray locs;

    size_t size() const { return self->relations[nid].get_num_blocks(); }

    // The relvars that have a fixed computes are inputs to the dag
    // and only have variables for locs
    bool fixed_computes() const { return self->info[nid].computes_set(); }

    int tensor_size() const; // TODO: implement properly

    void load_balance();
    void must_live_at();
    void inputs_live_at_computed_at();

    void set_minimum_locs();

    void set_constraints() {
      if(!fixed_computes()) {
        // constrain where the nodes can be computed
        load_balance();

        // Each block must live at where it is computed
        must_live_at();

        // The inputs for each block must live at
        //   wherever this guys gets computed at
        inputs_live_at_computed_at();
      } else {
        // For fixed nodes, we already know a minimum set of where things
        // must be computed, so constrain that
        set_minimum_locs();
      }
    }

    Placement* self;
    nid_t nid;
  };
  using relvar_ptr = std::unique_ptr<relvar_t>;

private:
  vector<relation_t> const& relations;
  vector<placement_t> const& info;
  vector<nid_t> _covering;
  bool _is_completely_covered;

  Gecode::Rnd rnd;
  Gecode::IntVar total_move;

  vector<relvar_ptr> vars;

  bool _set_covering();
  void _add(nid_t nid, std::set<nid_t>& so_far, int& num_added) const;

  void _set_branching();

  // For a node, get the non-no-op outputs of that node
  vector<nid_t> get_inputs(nid_t nid) const; // TODO: maybe cache?
public:
  virtual Gecode::Space* copy() {
    return new Placement(*this);
  }

  virtual Gecode::IntVar cost() const {
    return total_move;
  }
};

vector<placement_t> solve_placement(
  vector<relation_t> const& relations,
  int num_nodes,
  int num_cover);

}}

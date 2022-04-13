#pragma once

#include "../dag.h"
#include "../partition_info.h"

#include "../../../src/utils/cache_holder.h"

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include <vector>
#include <tuple>
#include <functional>

namespace bbts { namespace dag {

using std::vector;
using std::tuple;
using std::function;

// ints are used everywhere, but to make it
// clear what things the ints are referring to,
// some aliases
using nid_t  = int; // an id of a node
using dim_t  = int; // a dimension size
using rank_t = int; // a rank (a dimension label)

struct partition_options_t : public dag_t {
  Gecode::IntPropLevel _ipl;
  int _restart_scale;
  int _seed;
  int _num_workers;
  double _flops_per_time;
  double _bytes_per_time;
  int _min_cost;
  int _input_bytes_multiplier;
  int _output_bytes_multiplier;
  int _flops_multiplier;
  int _max_units;
  int _min_units;
  bool _disallow_barrier_reblock;
  int _search_compute_threads;
  int _search_restart_scale;
  int _search_time_per_cover;

public:
  partition_options_t(
    vector<node_t> const& dag,
    Gecode::IntPropLevel ipl,
    int restart_scale,
    int seed,
    int num_workers,
    double flops_per_time,
    double bytes_per_time,
    int min_cost,
    int input_bytes_multiplier,
    int output_bytes_multiplier,
    int flops_multiplier,
    int max_units,
    int min_units,
    bool disallow_barrier_reblock,
    int search_compute_threads,
    int search_restart_scale,
    int search_time_per_cover):
      dag_t(dag),
      _ipl(ipl),
      _restart_scale(restart_scale),
      _seed(seed),
      _num_workers(num_workers),
      _flops_per_time(flops_per_time),
      _bytes_per_time(bytes_per_time),
      _min_cost(min_cost),
      _input_bytes_multiplier(input_bytes_multiplier),
      _output_bytes_multiplier(output_bytes_multiplier),
      _flops_multiplier(flops_multiplier),
      _max_units(max_units),
      _min_units(min_units),
      _disallow_barrier_reblock(disallow_barrier_reblock),
      _search_compute_threads(search_compute_threads),
      _search_restart_scale(search_restart_scale),
      _search_time_per_cover(search_time_per_cover),
      _possible_parts(std::bind(&partition_options_t::_set_possible_parts, this))
  {}

  Gecode::IntPropLevel     ipl()                      const { return _ipl                      ; }
  int                      restart_scale()            const { return _restart_scale            ; }
  int                      seed()                     const { return _seed                     ; }
  int                      num_workers()              const { return _num_workers              ; }
  double                   flops_per_time()           const { return _flops_per_time           ; }
  double                   bytes_per_time()           const { return _bytes_per_time           ; }
  int                      min_cost()                 const { return _min_cost                 ; }
  int                      input_bytes_multiplier()   const { return _input_bytes_multiplier   ; }
  int                      output_bytes_multiplier()  const { return _output_bytes_multiplier  ; }
  int                      flops_multiplier()         const { return _flops_multiplier         ; }
  int                      max_units()                const { return _max_units                ; }
  int                      min_units()                const { return _min_units                ; }
  bool                     disallow_barrier_reblock() const { return _disallow_barrier_reblock ; }
  int                      search_compute_threads()   const { return _search_compute_threads   ; }
  int                      search_restart_scale()     const { return _search_restart_scale     ; }
  int                      search_time_per_cover()    const { return _search_time_per_cover    ; }

  vector<int> const& possible_parts() const {
    return _possible_parts();
  }
  vector<int> all_parts(dim_t dim) const;

  struct cost_t {
    cost_t(bool b):
      _is_no_op(b), _input_bytes(0), _output_bytes(0), _flops(0)
    {
      assert(_is_no_op);
    }

    cost_t(uint64_t i, uint64_t o, uint64_t f):
      _is_no_op(false), _input_bytes(i), _output_bytes(o), _flops(f)
    {}

    bool is_no_op() const { return _is_no_op; }

    uint64_t input_bytes()  const { assert(!is_no_op()); return _input_bytes;  }
    uint64_t output_bytes() const { assert(!is_no_op()); return _output_bytes; }
    uint64_t flops()        const { assert(!is_no_op()); return _flops;        }

  private:
    bool     const _is_no_op;
    uint64_t const _input_bytes;
    uint64_t const _output_bytes;
    uint64_t const _flops;
  };

  cost_t _cost(
    vector<vector<int>> const& inputs_bs,
    vector<int>         const& output_bs,
    vector<int>         const& flop_bs) const;
  int to_gecode_cost(cost_t const& c) const;

  struct cost_terms_t {
    int init;
    int input;
    int output;
    int compute;
  };
  cost_terms_t to_gecode_cost_terms(cost_t const& c) const;

  int get_node_cost(
    function<vector<int>(nid_t)> get_inc_part,
    nid_t nid,
    int num_assigned_workers) const;

  // get the cost for a node given the required incident parts.
  cost_t get_kernel_cost(
    // this function should get the partition for whatever comptue nodes necessary
    function<vector<int>(nid_t)> get_inc_part,
    // get the cost of this particular node with the required partitioning
    nid_t nid) const;
  int get_num_units(
    function<vector<int>(nid_t)> get_inc_part,
    nid_t nid) const;

  cost_t get_reblock_kernel_cost(
    vector<int> const& out_part,
    vector<int> const& inn_part,
    nid_t reblock_nid) const;
  cost_t get_join_kernel_cost(
    vector<int> const& inc_part,
    nid_t join_nid) const;
  cost_t get_agg_kernel_cost(
    vector<int> const& inc_part,
    nid_t agg_nid) const;
  cost_t get_input_kernel_cost(
    vector<int> const& inc_part,
    nid_t input_nid) const;

  vector<int> get_local_dims(
      vector<int> const& inc_part,
      nid_t nid) const;
  vector<int> get_local_dims(
    function<vector<int>(nid_t)> get_inc_part,
    nid_t nid) const
  {
    return get_local_dims(get_inc_part(nid), nid);
  }

  vector<vector<int>> all_partitions(nid_t nid, int m) const;
  vector<vector<int>> all_partitions(nid_t nid) const {
    return all_partitions(nid, max_units());
  }

  // get the max allowable partition
  vector<int> max_partition(nid_t nid, int m) const;
  vector<int> max_partition(nid_t nid) const {
    return max_partition(nid, max_units());
  }

  // given that each node has it's max partition up to the number
  // of workers, figure out how long the whole thing will take, doing
  // one node after the other
  int upper_bound_time() const;
private:
  mutable std::unordered_map<int, vector<int>> _all_parts_cache;

  cache_holder_t<vector<int>> _possible_parts;

  vector<int> _set_possible_parts();
};

struct Partition : public Gecode::IntMinimizeSpace {
  Partition(partition_options_t const& opt);

  Partition(Partition &other);

  virtual void print(std::ostream& os) const;

  vector<int> get_partition(nid_t nid)         const;
  int         get_start(nid_t nid)             const;
  int         get_duration(nid_t nid)          const;
  int         get_worker(nid_t nid)            const;
  int         get_unit(nid_t nid)              const;

private:
  struct pode_t {
    pode_t(Partition* self, nid_t nid, int upper_limit);
    pode_t(Partition* new_self, pode_t& other);

    // virtual final means no one else can override this function
    virtual void set_base_constraint() final;

    // each child class must implement these
    virtual void set_constraints() = 0;

    // Is this a no op? then that implies a bunch of things are zero
    void _set_no_op(Gecode::BoolVar& is_no_op);

    Gecode::IntVar start;           // this can start when inputs are available
    Gecode::IntVar end;             // end = start + duration
    Gecode::IntVar duration;        // duration == (unit / worker) * kernel_duration
    Gecode::IntVar worker;          // the number of workers

    Gecode::IntVar unit;            // number of parallel units of work
    Gecode::IntVar kernel_duration; // how long a unit takes

    // When a node has enough information, unit and kerenl_duration are set
    void _set_unit_and_kernel_duration(int fixed_unit, int fixed_kernel_duration);

    Partition* self;
    nid_t nid;

    int rank() const { return self->opt[nid].dims.size(); }
  };
  struct pagg_t : pode_t {
    pagg_t(Partition* self, nid_t nid, int upper_limit);
    pagg_t(Partition* new_self, pagg_t& other);
    Gecode::BoolVar is_no_op;
    void propagate_is_no_op();
    void when_partition_info_set();
    void set_constraints() override {
      propagate_is_no_op();
      when_partition_info_set();
    }

    Gecode::IntVarArgs local_partition();
  };
  struct preblock_t : pode_t {
    preblock_t(Partition* self, nid_t nid, int upper_limit);
    preblock_t(Partition* new_self, preblock_t& other);
    Gecode::BoolVar is_no_op;
    void propagate_is_no_op();
    void when_partition_info_set();
    void disallow_barrier_reblock();
    void set_constraints() override {
      propagate_is_no_op();
      when_partition_info_set();
      if(self->opt.disallow_barrier_reblock()) {
        disallow_barrier_reblock();
      }
    }

    Gecode::IntVarArgs local_partition_above();
    Gecode::IntVarArgs local_partition_below();
  };
  struct pjoin_t: pode_t {
    pjoin_t(Partition* self, nid_t nid, int upper_limit);
    pjoin_t(Partition* new_self, pjoin_t& other);

    Gecode::IntVarArray partition;

    void when_partition_info_set();
    void match_downs();
    void set_constraints() override {
      when_partition_info_set();
      match_downs();
    }
  };
  struct pinput_t : pode_t {
    pinput_t(Partition* self, nid_t nid, int upper_limit);
    pinput_t(Partition* new_self, pinput_t& other);

    Gecode::IntVarArray partition;

    void propagate_prefer_no_reblock();
    void propagate_no_cost_input();
    void when_partition_info_set();
    void set_constraints() override {
      propagate_prefer_no_reblock();
      propagate_no_cost_input();
      when_partition_info_set();
    }
  };

  using pode_ptr = std::unique_ptr<pode_t>;

private:
  partition_options_t const& opt;

  Gecode::Rnd rnd;

  Gecode::IntVar makespan;

  vector<pode_ptr> vars;

  void _cumulative(
    int                   capacity,
    Gecode::IntVarArgs    const& start,
    Gecode::IntVarArgs    const& duration,
    Gecode::IntVarArgs    const& end,
    Gecode::IntVarArgs    const& resource_usage,
    Gecode::IntPropLevel  ipl = Gecode::IPL_DEF);

  void _set_branching();
public:
  virtual Gecode::Space* copy() {
    return new Partition(*this);
  }

  virtual Gecode::IntVar cost() const {
    return makespan;
  }
};

// For each node, get the run info by
// unleasing Gecode with Partition
vector<partition_info_t> run_partition(partition_options_t const& opt);

}}

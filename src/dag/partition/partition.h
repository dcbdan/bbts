#pragma once

#include "../node.h"

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include <map>

namespace bbts { namespace dag {

using std::vector;
using std::tuple;

// ints are used everywhere, but to make it
// clear what things the ints are referring to,
// some aliases
using nid_t  = int; // an id of a node
using dim_t  = int; // a dimension size
using rank_t = int; // a rank (a dimension label)

class partition_options_t {
  vector<node_t> dag;

  //  _restart_scale("restart-scale","scale factor for restart sequence",150),
  //  _seed("seed","random number generator seed",1U),
  //  _dag_file("dag-file", "File containing the dag to partition", "dag/matmul.dag"),
  //  _num_workers("num-workers", "Number of workers", 24),
  //  _flops_per_time("flops-per-time", "Number of flops per unit of time", 1e8),
  //  _min_cost("min-cost", "defaults to number of workers", -1),
  //  _breadth_order("breadth-order", "Additional constraint on order of nodes", false),
  //  _cover("cover", "Whether or not to use the cover algorithm", false),
  //  _cover_size("cover-size", "Number of nodes to consider at a time in iterative solve", 20),
  //  _search_compute_threads("search-compute-threads", "Number of threads for gecode to search with", 24),
  //  _search_restart_scale("search-restart-scale", "Restart scale param", Search::Config::slice),
  //  _search_time_per_cover("search-time-per-cover", "How long each iteration can take, ms", 4000),
  //  _output_file("output-file", "write the blockings here", ""),
  //  _usage_file("usage-file", "write out [(start,end,thread,label)] to this file", "")

  Gecode::IntPropLevel _ipl;
  int _restart_scale;
  int _seed;
  int _num_workers;
  double _flops_per_time;
  int _min_cost;
  int _breadth_order;
  bool _cover;
  int _cover_size;
  int _search_compute_threads;
  int _search_restart_scale;
  int _search_time_per_cover;

public:
  partition_options_t(vector<node_t> const& dag):
    dag(dag),
    _ipl(Gecode::IPL_DEF),
    _restart_scale(150),
    _seed(1),
    _num_workers(24),
    _flops_per_time(1e8),
    _min_cost(-1),
    _breadth_order(false),
    _cover(false),
    _cover_size(20),
    _search_compute_threads(24),
    _search_restart_scale(Gecode::Search::Config::slice),
    _search_time_per_cover(400)
  {
    // cache these things
    _set_all_partitions_per_node();
    _set_dag_orders();
  }

  int                      get_restart_scale()      const;
  int                      seed()                   const;
  Gecode::IntPropLevel     ipl()                    const;
  vector<node_t> const&    get_dag()                const;
  int                      get_num_workers()        const;
  double                   get_flops_per_time()     const;
  vector<int>              get_all_blocks()         const;
  int                      get_min_cost()           const;
  bool                     breadth_order()          const;
  bool                     cover()                  const;
  int                      cover_size()             const;
  int                      search_compute_threads() const;
  int                      search_restart_scale()   const;
  double                   search_time_per_cover()  const;

  // Computing things over the dag

  // take the product of dims and turn it into an integer that corresponds to flops
  int raw_flops(vector<int> const& dims) const;

  // compute cost = min_cost + raw_flops
  int get_compute_cost(int raw_flops_cost) const;
  int get_compute_cost(vector<int> const& dims) const;

  // get dag orders
  vector<nid_t> const& inputs() const;
  vector<nid_t> const& depth_dag_order() const;
  vector<nid_t> const& breadth_dag_order() const;
  vector<nid_t> const& super_depth_dag_order() const;
  vector<nid_t> const& super_breadth_dag_order() const;

  // return {input} or {[reblocks], join, agg}.
  vector<nid_t> super(nid_t nid) const;

  // get just joins and inputs
  vector<nid_t> get_compute_nids() const;
  vector<nid_t> get_compute_ups(nid_t) const;

  // each node has an associated compute node.
  // reblocks and aggs are attached to a join,
  // inputs are inputs.
  nid_t get_compute_nid(nid_t nid) const;

  // a partition of all 1s
  vector<dim_t> single_partition(nid_t nid) const;
  // the largest partition for a node
  vector<dim_t> max_partition(nid_t nid) const;
  // all incident partitions a node can have
  vector<vector<dim_t>> const& all_partitions(nid_t id) const;
  // given an index to all_partitions(nid), return the corresponding partition
  // so all_partitions(nid)[which] == get_which_partition(nid, which)
  vector<dim_t> const& get_which_partition(nid_t nid, int which) const;

  // return all possible number of blocks for this dimension
  vector<dim_t> all_blocks(dim_t dim) const;

  // Compute the time it takes to get to each node.
  // There are no resource limitations or anything.
  // This can serve as the minimum time for each node given resource
  // limitations.
  vector<int> time_to_completion(
    // for each node, how long that computation takes
    vector<int> const& times) const;

  vector<vector<dim_t>> get_random_full_dag_partitioning() const;
  vector<vector<dim_t>> get_max_full_dag_partitioning() const;

  // for each partition of nid, get the kernel size
  vector<dim_t> get_kernel_inc_dims(vector<dim_t> const& parts, nid_t nid) const;

  // filter out aggs from xs (if there are even any aggs)
  vector<int> get_out(vector<int> const& xs, nid_t nid) const;

  // filter out the non aggs (keeps) from xs
  vector<int> get_agg(vector<int> const& xs, nid_t nid) const;

  // given the join-incident dims, get the output of the reblock
  vector<int> get_reblock_out(vector<int> const& join_inc, nid_t reblock_id) const;

  vector<int> get_out_from_compute(vector<int> const& inc, nid_t nid) const;

  // given the join-incident partition, get the (computation time, num workers) pairs
  // (assuming a computation will happen!)
  // This function should work for any node type
  tuple<int, int> get_est_duration_worker_pair(vector<int> const& inc_part, nid_t nid) const;

  tuple<int, int> get_duration_worker_pair(
    vector<vector<int>> const& all_parts,
    nid_t nid) const;
  tuple<int, int> get_duration_worker_pair(
    std::function<vector<int>(nid_t)> f,
    nid_t nid) const;

private:
  // some helper functions
  void _depth_dag_order_add_to_ret(
    vector<nid_t>& counts,
    vector<nid_t>& ret,
    nid_t id) const;
  void _super_depth_dag_order_add_to_ret(
    std::map<nid_t, int>& counts,
    vector<nid_t>& ret,
    nid_t id) const;
  vector<vector<dim_t>> _get_partitioning(
    std::function<vector<dim_t>(nid_t)> f) const;

  // a cache of all possible partitions of each node,
  // sorted from most number of workers to least
  vector<vector<vector<dim_t>>> _all_partitions_per_node;
  void _set_all_partitions_per_node();

  vector<nid_t> _breadth_dag_order;
  vector<nid_t> _depth_dag_order;
  vector<nid_t> _super_breadth_dag_order;
  vector<nid_t> _super_depth_dag_order;
  void _set_dag_orders();
  vector<nid_t> _inputs;
};

///////////////////////////////////////////////////////////////////////////////

struct partition_init_t {
  partition_init_t(int n):
    start_min(n),      start_max(n),
    duration_min(n),   duration_max(n),
    worker_min(n),     worker_max(n),
    partitions_min(n), partitions_max(n),
    min_possible_time(0), max_possible_time(0)
  {}

  vector<int> start_min,      start_max;
  vector<int> duration_min,   duration_max;
  vector<int> worker_min,     worker_max;
  vector<int> partitions_min, partitions_max;

  int min_possible_time, max_possible_time;

  Gecode::IntVar start(Gecode::Space& home, nid_t nid) const {
    return Gecode::IntVar(home, start_min[nid], start_max[nid]);
  }
  Gecode::IntVar duration(Gecode::Space& home, nid_t nid) const {
    return Gecode::IntVar(home, duration_min[nid], duration_max[nid]);
  }
  Gecode::IntVar end(Gecode::Space& home, nid_t nid) const {
    return Gecode::IntVar(home, 0, max_possible_time);
  }
  Gecode::IntVar worker(Gecode::Space& home, nid_t nid) const {
    return Gecode::IntVar(home, worker_min[nid], worker_max[nid]);
  }
  Gecode::IntVar partitions(Gecode::Space& home, nid_t nid) const {
    return Gecode::IntVar(home, partitions_min[nid], partitions_max[nid]);
  }
};

struct Partition : public Gecode::IntMinimizeSpace {
  // Set up the space for use with optimization. The cover is set to the
  // initial covering
  Partition(partition_options_t const& opt0, partition_init_t const& init);

  // Take the pervious space fix previously computed values and increment
  // the cover
  Partition(Partition const& other, partition_init_t const& init);

  static partition_init_t build_init(partition_options_t const& opt);

  virtual void print(std::ostream& os) const;

protected:
  partition_options_t const& opt;

  Gecode::Rnd rnd;

  // We're trying to minimize this guy
  Gecode::IntVar makespan;

  Gecode::IntVarArray start;
  Gecode::IntVarArray duration;
  Gecode::IntVarArray end;
  Gecode::IntVarArray worker;

  // For each input and join node we have a partition
  // For all slots that aren't inputs or joins, just set the domain equal to {0} as it
  // won't be used.
  Gecode::IntVarArray partitions;
  // ^ given a compute identifier nid, partitions[nid] specifies
  //   which partition in opt.all_partitions(nid) the current partition has.

  int cover_to;

public:
  Partition(Partition& other):
    Gecode::IntMinimizeSpace(other),
    opt(other.opt),
    rnd(other.rnd),
    cover_to(other.cover_to)
  {
    makespan.update  (  *this, other.makespan    );

    start.update     (  *this, other.start       );
    duration.update  (  *this, other.duration    );
    end.update       (  *this, other.end         );
    worker.update    (  *this, other.worker      );

    partitions.update(  *this, other.partitions  );
  }

  virtual Gecode::Space* copy(void) {
    return new Partition(*this);
  }

  virtual Gecode::IntVar cost() const {
    return makespan;
  }

  bool covers_all() const {
    return _get_nids(cover_to).size() >= opt.get_dag().size();
  }

  Gecode::IntVar const& get_partition(nid_t nid) const {
    return partitions[nid];
  }

  // print a file where each line contains
  // (start,finish,worker,label)
  void write_usage(std::ostream& out) const;

private:
  void assign_join_agg(nid_t join_nid);
  void assign_reblock(nid_t reblock_nid);

  int select_partition(nid_t nid) const;

  void _cumulative(
    int                capacity,
    Gecode::IntVarArgs  const& start,
    Gecode::IntVarArgs  const& duration,
    Gecode::IntVarArgs  const& end,
    Gecode::IntVarArgs  const& resource_usage,
    Gecode::IntPropLevel       ipl = Gecode::IPL_DEF);

  vector<nid_t> _get_nids(int n) const;

  // These propagations are for all values of voer to
  void _init_propagations();

  // use cover_to to specify the problem and the branches
  int _cover_propagations_and_branches(int n);
};

}}


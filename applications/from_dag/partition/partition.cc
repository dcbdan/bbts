#include "partition.h"

#include "../misc.h"

#include <unordered_map>
#include <numeric>
#include <set>

#define DCB_BEFORE_DYNAMIC(x) //std::cout << __LINE__ << " " << x << std::endl
#define DCB_ACCESS_VAR(x)     //std::cout << __LINE__ << " " << x << std::endl;
#define DCB_P_CONSTRUCTOR(x)  //std::cout << __LINE__ << " " << x << std::endl;
#define DCB01(x)              //std::cout << __LINE__ << " " << x << std::endl;
#define DCB02(x)              //std::cout << x << std::endl;
#define DCB_COVER(x)          //std::cout << __LINE__ << " " << x << std::endl;

namespace bbts { namespace dag {

using namespace Gecode;

// inn   out   ret  ceiling(inn / out)
// 1     1     1     1
// 2     1     2     2
// 1     2     1     1
// 3     2     2     2
// 2     3     2     1
// 4     3
int get_max_num_inputs_est(int inn, int out) {
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

vector<vector<int>> cartesian(
  vector<vector<int> > const& vs)
{
  // assuming all vs have a size greater than zero

  vector<int> max;
  for(auto const& v: vs) {
    max.push_back(v.size());
  }
  indexer_t indexer(max);

  vector<vector<int>> ret;
  do {
    vector<int> next_val;
    next_val.reserve(vs.size());
    for(int which_v = 0; which_v != vs.size(); ++ which_v) {
      next_val.push_back(vs[which_v][indexer.idx[which_v]]);
    }
    ret.push_back(next_val);
  } while(indexer.increment());

  return ret;
}

vector<int> partition_options_t::_set_possible_parts()
{
  vector<int> ret;
  ret.reserve(40);
  // just powers of 2s and multiples of 12,
  // early multiples of 3
  for(int x: {1,2,3,4,6,8,9,12,15,18,21,24,32,36,48,
              60,64,72,84,96,108,120,132,144})
  {
    if(x <= num_workers()) {
      ret.push_back(x);
    } else {
      return ret;
    }
  }
  return ret;
}

vector<int> partition_options_t::all_parts(dim_t dim) const {
  if(_all_parts_cache.count(dim) == 0) {
    vector<int> val;
    val.reserve(possible_parts().size());
    for(int const& p: possible_parts()) {
      if(dim % p == 0) {
        val.push_back(p);
      }
    }
    _all_parts_cache[dim] = val;
    return val;
  }
  return _all_parts_cache[dim];
}

vector<vector<int>>
partition_options_t::all_partitions(nid_t nid, int m) const
{
  node_t const& node = dag[nid];

  vector<vector<int>> options;
  options.reserve(node.dims.size());
  for(int r = 0; r != node.dims.size(); ++r) {
    options.push_back(all_parts(node.dims[r]));
  }

  // get all possible partitions.
  // A partition is not possible is the number of units
  // is more than "m"
  auto xs = cartesian(options);
  vector<vector<int>> ret;
  for(auto const& x: xs) {
    if(product(x) <= m) {
      ret.push_back(x);
    }
  }
  return ret;
}

vector<int>
partition_options_t::max_partition(nid_t nid, int m) const
{
  auto xs = all_partitions(nid, m);
  int best = product(xs[0]);
  auto ret = xs[0];
  for(int i = 1; i < xs.size(); ++i) {
    int score = product(xs[i]);
    if(score > best) {
      score = best;
      ret = xs[i];
    }
  }
  return ret;
}

uint64_t product_to_uint64(vector<int> const& xs) {
  uint64_t ret = 1;
  for(int const& x: xs) {
    assert(x >= 0);
    ret *= x;
  }
  return ret;
}

partition_options_t::cost_t
partition_options_t::_cost(
  vector<vector<int>> const& inputs_bs,
  vector<int>         const& output_bs,
  vector<int>         const& flop_bs) const
{
  uint64_t input_total = 0;
  for(auto const& bs: inputs_bs) {
    input_total += product_to_uint64(bs);
  }

  return partition_options_t::cost_t(
    input_total,
    product_to_uint64(output_bs),
    product_to_uint64(flop_bs)
  );
}

partition_options_t::cost_terms_t
partition_options_t::to_gecode_cost_terms(partition_options_t::cost_t const& c) const {
  if(c.is_no_op()) {
    return cost_terms_t {
      .init    = 0,
      .input   = 0,
      .output  = 0,
      .compute = 0
    };
  }
  auto ceiling_div = [](uint64_t x, int y) {
    // https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    if(x == 0) { return 0; }
    assert(y > 0);
    uint64_t ret = (1 + ((x - 1) / y)); // if x != 0

    // make sure the return value should is less than, say, 2^30.
    assert(ret < (uint64_t(1) << 30));
    return int(ret);
  };

  return cost_terms_t {
    .init = min_cost(),
    .input =
      input_bytes_multiplier()  * ceiling_div(c.input_bytes(),  bytes_per_time()),
    .output =
      output_bytes_multiplier() * ceiling_div(c.output_bytes(), bytes_per_time()),
    .compute =
      flops_multiplier()        * ceiling_div(c.flops(),        flops_per_time())
  };
}

int partition_options_t::to_gecode_cost(partition_options_t::cost_t const& c) const {
  cost_terms_t r = to_gecode_cost_terms(c);
  return r.init + r.input + r.output + r.compute;
}

partition_options_t::cost_t
partition_options_t::get_reblock_kernel_cost(
  vector<int> const& out_part,
  vector<int> const& inn_part,
  nid_t reblock_nid) const
{
  node_t const& node = dag[reblock_nid];
  vector<int> const& dims = node.dims;

  assert(out_part.size() == dims.size() &&
         inn_part.size() == dims.size());

  assert(node.type == node_t::node_type::reblock);

  if(out_part == inn_part) {
    // this is a no op, nothing to do
    return partition_options_t::cost_t(true);
  }

  vector<vector<int>> inputs_bs;
  vector<int> output_bs, flop_bs;

  vector<int> inn_shape;
  vector<int> out_shape;
  inn_shape.reserve(dims.size());
  out_shape.reserve(dims.size());
  for(int i = 0; i != inn_part.size(); ++i) {
    inn_shape.push_back(dims[i] / inn_part[i]);
    out_shape.push_back(dims[i] / out_part[i]);
  }

  output_bs = out_shape;

  flop_bs = out_shape;

  int max_num_inputs = 1;
  for(int r = 0; r != dims.size(); ++r) {
    max_num_inputs *= get_max_num_inputs_est(inn_part[r], out_part[r]);
  }
  inputs_bs.push_back(inn_shape);
  inputs_bs[0].push_back(max_num_inputs);

  return _cost(inputs_bs, output_bs, flop_bs);
}

partition_options_t::cost_t
partition_options_t::get_join_kernel_cost(
  vector<int> const& inc_part,
  nid_t join_nid) const
{
  vector<vector<int>> inputs_bs;
  vector<int> output_bs, flop_bs;

  node_t const& node = dag[join_nid];
  assert(node.type == node_t::node_type::join);

  flop_bs = get_local_dims(inc_part, join_nid);
  output_bs = get_out(flop_bs, join_nid);
  for(nid_t child: node.downs) {
    inputs_bs.push_back(get_out_for_input(flop_bs, join_nid, child));
  }

  return _cost(inputs_bs, output_bs, flop_bs);
}

partition_options_t::cost_t
partition_options_t::get_agg_kernel_cost(
  vector<int> const& inc_part,
  nid_t agg_nid) const
{
  nid_t owner_nid = get_part_owner(agg_nid);

  vector<vector<int>> inputs_bs;
  vector<int> output_bs, flop_bs;

  node_t const& node = dag[agg_nid];
  assert(node.type == node_t::node_type::agg);

  auto inc_local = get_local_dims(inc_part, owner_nid);

  int num_to_agg = product(get_agg(inc_part, owner_nid));
  if(num_to_agg == 1) {
    return partition_options_t::cost_t(true);
  }

  output_bs = get_out(inc_local, owner_nid);

  inputs_bs.push_back(output_bs);
  inputs_bs[0].push_back(num_to_agg);

  flop_bs = inputs_bs[0];

  return _cost(inputs_bs, output_bs, flop_bs);
}

partition_options_t::cost_t
partition_options_t::get_input_kernel_cost(
  vector<int> const&,
  nid_t) const
{
  return partition_options_t::cost_t(true);
}

partition_options_t::cost_t
partition_options_t::get_kernel_cost(
  function<vector<int>(nid_t)> get_inc_part,
  nid_t nid) const
{
  node_t const& node         = dag[nid];

  if(node.type == node_t::node_type::input) {
    DCB01("input");
    return get_input_kernel_cost(get_inc_part(nid), nid);
  } else
  if(node.type == node_t::node_type::join) {
    DCB01("join");
    return get_join_kernel_cost(get_inc_part(nid), nid);
  } else
  if(node.type == node_t::node_type::agg) {
    DCB01("agg");
    return get_agg_kernel_cost(
      get_inc_part(node.downs[0]),
      nid);
  } else
  if(node.type == node_t::node_type::reblock) {
    DCB01("reblock");
    nid_t owner_nid   = get_part_owner(nid);
    nid_t owner_below = get_part_owner(node.downs[0]);

    auto above_part_inc = get_inc_part(owner_nid);
    auto below_part_inc = get_inc_part(owner_below);

    auto out_part = get_out_for_input(above_part_inc, owner_nid, nid);
    auto inn_part = get_out(below_part_inc, owner_below);

    return get_reblock_kernel_cost(out_part, inn_part, nid);
  } else {
    assert(false);
    return partition_options_t::cost_t(true);
  }
}

vector<int> partition_options_t::get_local_dims(
    vector<int> const& inc_part,
    nid_t nid) const
{
  node_t const& node = dag[nid];

  assert(node.dims.size() == inc_part.size());

  vector<int> ret;
  ret.reserve(inc_part.size());
  for(int r = 0; r != inc_part.size(); ++r) {
    assert(node.dims[r] % inc_part[r] == 0);
    ret.push_back(node.dims[r] / inc_part[r]);
  }
  return ret;
}

int partition_options_t::upper_bound_time() const {
  std::unordered_map<int, vector<int>> parts;

  // This will get called for only for part_owner nodes..
  // And in this case, for all part owner nodes.
  function<vector<int>(nid_t)> get_inc_part = [&](nid_t nid) {
    if(parts.count(nid) == 0) {
      parts[nid] = max_partition(nid, num_workers());
    }
    return parts[nid];
  };

  int total = 0;
  for(nid_t nid: breadth_dag_order()) {
    // there is no need to call get_node_cost since
    // the all kernels for this guy will happen all at once.
    // (the max units is set to num_workers() in get_inc_part)
    total += to_gecode_cost(get_kernel_cost(get_inc_part, nid));
  }
  return total;
}

int partition_options_t::get_node_cost(
  function<vector<int>(nid_t)> get_inc_part,
  nid_t nid,
  int num_assigned_workers) const
{
  int kernel_cost = to_gecode_cost(get_kernel_cost(get_inc_part, nid));
  int num_units = get_num_units(get_inc_part, nid);
  assert(num_units % num_assigned_workers == 0);
  int num_calls = num_units / num_assigned_workers;
  return kernel_cost * num_calls;
}

int partition_options_t::get_num_units(
  function<vector<int>(nid_t)> get_inc_part,
  nid_t nid) const
{
  return product(
    get_node_incident(get_inc_part(get_part_owner(nid)), nid));
}

Partition::pode_t::pode_t(
  Partition* self_, nid_t nid_, int upper_limit):
    self(self_), nid(nid_), fixed(false)
{
  start    = IntVar(*self, 0, upper_limit);
  duration = IntVar(*self, 0, upper_limit);

  // set end directly
  end = expr(*self, start + duration);

  worker   = IntVar(*self, 0, self->opt.num_workers());

  unit     = IntVar(*self, 0, self->opt.num_workers() * self->opt.max_units());

  kernel_duration = IntVar(*self, 0, upper_limit);
}

void Partition::pode_t::set_base_constraint() {
  // this node can start after child nodes
  node_t const& node = self->opt[nid];
  for(nid_t nid: node.downs) {
    pode_t& child = *(self->vars[nid]);
    rel(*self, child.end <= start, self->opt.ipl());
  }

  // the makespan is done when this node is done
  rel(*self, end <= self->makespan, self->opt.ipl());

  // the duration is just the kernel duration called however many times
  IntVar call(*self, 0, self->opt.num_workers() * self->opt.max_units());
  rel(*self, duration == call * kernel_duration, self->opt.ipl());

  // the number of workers determines how many times it must be called

  ////////////////////////////////////////////////////////
  //rel(*self, unit == worker);
  ////////////////////////////////////////////////////////

  rel(*self, unit == call * worker);
  rel(*self, (unit == 0) >> (call == 0));
  rel(*self, (unit == 0) >> (worker == 0));
  rel(*self, (unit == 0) == (kernel_duration == 0));


  if(self->opt.min_units() > 0) {
    rel(*self, (unit > 0) >> (unit >= (self->opt.min_units())));
    //rel(*self, (unit > 0) >> (worker >= self->opt.min_units()));
  }

  // this should be held by the initial domain, but just in case
  rel(*self, unit <= self->opt.max_units());
}

Partition::pagg_t::pagg_t(
  Partition* self, nid_t nid, int upper_limit):
    Partition::pode_t(self, nid, upper_limit)
{
  is_no_op = BoolVar(*self, 0, 1);
}

Partition::preblock_t::preblock_t(
  Partition* self, nid_t nid, int upper_limit):
    Partition::pode_t(self, nid, upper_limit)
{
  is_no_op = BoolVar(*self, 0, 1);
}

#define CONSTRUCTOR_PARTITION                                          \
  auto const& opt = self->opt;                                         \
  node_t const& node = opt[nid];                                       \
  partition = IntVarArray(*self, node.dims.size());                    \
  for(int i = 0; i != node.dims.size(); ++i) {                         \
    partition[i] = IntVar(*self, IntSet(opt.all_parts(node.dims[i]))); \
  }

Partition::pinput_t::pinput_t(
  Partition* self, nid_t nid, int upper_limit):
    Partition::pode_t(self, nid, upper_limit)
{
  CONSTRUCTOR_PARTITION
}

Partition::pjoin_t::pjoin_t(
  Partition* self, nid_t nid, int upper_limit):
    Partition::pode_t(self, nid, upper_limit)
{
  CONSTRUCTOR_PARTITION
}

Partition::Partition(partition_options_t const& opt):
  Partition::Partition(opt, vector<nid_t>{})
{}

Partition::Partition(partition_options_t const& opt0, Partition const& other):
  Partition::Partition(opt, other.covered())
{
  // For each item covered by other, fix the values to whatever was found
  for(nid_t const& nid: other.covered()) {
    vars[nid]->fix(*other.vars[nid]);
  }
}

Partition::Partition(partition_options_t const& opt0, vector<nid_t> must_cover_these):
  IntMinimizeSpace(),
  opt(opt0),
  rnd(opt0.seed())
{
  DCB_P_CONSTRUCTOR("Z");
  _set_covering(must_cover_these);

  DCB_P_CONSTRUCTOR("A");
  int upper_limit = 10*opt.upper_bound_time();
  DCB_P_CONSTRUCTOR("B");
  DCB02("UPPER LIMIT OF " << upper_limit);

  makespan = IntVar(*this, 0, upper_limit);
  DCB_P_CONSTRUCTOR("C");

  vars = vector<pode_ptr>(opt.size());
  for(auto& ptr: vars) {
    ptr = nullptr;
  }

  for(nid_t const& nid: covered()) {
    DCB_P_CONSTRUCTOR("init vars " << nid);
    node_t const& node = opt[nid];
    if(node.type == node_t::node_type::input) {
      vars[nid] = std::unique_ptr<pode_t>(new pinput_t(this, nid, upper_limit));
    } else
    if(node.type == node_t::node_type::join) {
      vars[nid] = std::unique_ptr<pode_t>(new pjoin_t(this, nid, upper_limit));
    } else
    if(node.type == node_t::node_type::agg) {
      vars[nid] = std::unique_ptr<pode_t>(new pagg_t(this, nid, upper_limit));
    } else
    if(node.type == node_t::node_type::reblock) {
      vars[nid] = std::unique_ptr<pode_t>(new preblock_t(this, nid, upper_limit));
    } else {
      throw std::runtime_error("::Partition should not reach");
    }
  }
  DCB_P_CONSTRUCTOR("D");

  // Set the dag ordering and the makespan and wtvr else
  // the base constraint does
  for(nid_t const& nid: covered()) {
    vars[nid]->set_base_constraint();
  }
  DCB_P_CONSTRUCTOR("E");

  // Set the cumulative constriant
  IntVarArgs start;
  IntVarArgs duration;
  IntVarArgs end;
  IntVarArgs worker;
  for(nid_t const& nid: covered()) {
    pode_t& p = *vars[nid];
    start    << p.start;
    duration << p.duration;
    end      << p.end;
    worker   << p.worker;
  }
  DCB_P_CONSTRUCTOR("F");

  _cumulative(opt.num_workers(), start, duration, end, worker);

  // For each node, set the partition related constraints specific to that node
  for(nid_t const& nid: covered()) {
    vars[nid]->set_constraints();
  }

  // Now that all the variables and the constraints on those variables are set up,
  // branching needs to happen.
  _set_branching();
  DCB_P_CONSTRUCTOR("G");
}

Partition::Partition(Partition& other):
  IntMinimizeSpace(other),
  opt(other.opt),
  rnd(other.rnd),
  _covered(other._covered)
{
  DCB_COVER("ENTERED Partition::Partition(Partition& other)");

  makespan.update(*this, other.makespan);

  vars = vector<pode_ptr>(opt.size());
  for(auto& ptr: vars) {
    ptr = nullptr;
  }

  // static cast should also be valid, right?
  for(nid_t const& nid: covered()) {
    DCB_COVER("nid: " << nid);
    node_t const& node = opt[nid];
    if(node.type == node_t::node_type::input) {
      DCB_BEFORE_DYNAMIC("A");
      vars[nid] = pode_ptr(new pinput_t(this,
        dynamic_cast<pinput_t&>(*other.vars[nid])));
    } else
    if(node.type == node_t::node_type::join) {
      DCB_BEFORE_DYNAMIC("B");
      vars[nid] = pode_ptr(new pjoin_t(this,
        dynamic_cast<pjoin_t&>(*other.vars[nid])));
    } else
    if(node.type == node_t::node_type::agg) {
      DCB_BEFORE_DYNAMIC("C");
      vars[nid] = pode_ptr(new pagg_t(this,
        dynamic_cast<pagg_t&>(*other.vars[nid])));
    } else
    if(node.type == node_t::node_type::reblock) {
      DCB_BEFORE_DYNAMIC("D");
      vars[nid] = pode_ptr(new preblock_t(this,
        dynamic_cast<preblock_t&>(*other.vars[nid])));
    } else {
      assert(false);
    }
  }
  DCB_COVER("EXITED Partition::Partition(Partition& other)");
}

Partition::pode_t::pode_t(Partition* new_self, Partition::pode_t& other):
  self(new_self),
  nid(other.nid),
  fixed(other.fixed)
{
  start.update           ( *self, other.start           );
  end.update             ( *self, other.end             );
  duration.update        ( *self, other.duration        );
  worker.update          ( *self, other.worker          );
  unit.update            ( *self, other.unit            );
  kernel_duration.update ( *self, other.kernel_duration );
}

Partition::pagg_t::pagg_t(Partition* new_self, Partition::pagg_t& other):
  Partition::pode_t(new_self, other)
{
  is_no_op.update(*self, other.is_no_op);
}

Partition::preblock_t::preblock_t(Partition* new_self, Partition::preblock_t& other):
  Partition::pode_t(new_self, other)
{
  is_no_op.update(*self, other.is_no_op);
}

Partition::pinput_t::pinput_t(Partition* new_self, Partition::pinput_t& other):
  Partition::pode_t(new_self, other)
{
  partition.update(*self, other.partition);
}

Partition::pjoin_t::pjoin_t(Partition* new_self, Partition::pjoin_t& other):
  Partition::pode_t(new_self, other)
{
  partition.update(*self, other.partition);
}

void Partition::_cumulative(
  int                capacity,
  IntVarArgs  const& _start,
  IntVarArgs  const& _duration,
  IntVarArgs  const& _end,
  IntVarArgs  const& _resource_usage,
  IntPropLevel       ipl)
{
  IntArgs zeros(vector<int>(_start.size(), 0));
  IntArgs capacities(vector<int>(1, capacity));
  cumulatives(*this,
    zeros, _start, _duration, _end, _resource_usage, capacities, true, ipl);
}

void Partition::print(std::ostream& os) const {
  int total = 0;
  for(nid_t const& nid: covered()) {
    auto& p = *vars[nid];
    total += p.worker.min() * p.duration.min();
  }
  float utilization = 100.0 * (total / (1.0*opt.num_workers()*makespan.max()));

  os << "Partition makespan, utilization:  " << makespan.max() << ",  ";
  os << utilization << "%" << std::endl;
}

void Partition::pagg_t::propagate_is_no_op()
{
  auto const& opt = self->opt;
  node_t const& node = opt[nid];

  nid_t join_nid = node.downs[0];
  node_t const& join_node = opt[join_nid];
  DCB_BEFORE_DYNAMIC("PAGG PROP");
  Partition::pjoin_t& pjoin = dynamic_cast<Partition::pjoin_t&>(*self->vars[join_nid]);

  BoolVarArgs args;
  for(auto const& which: join_node.aggs) {
    BoolVar v = expr(*self, pjoin.partition[which] == 1);
    args << v;
  }

  // This node is a no op if and only if all agg incident partitions
  // have size 1.
  rel(*self, BOT_AND, args, is_no_op);

  // If this is a no op, set a bunch of stuff to zero
  Partition::pode_t::_set_no_op(is_no_op);
}

IntVarArgs
Partition::preblock_t::local_partition_above()
{
  auto const& opt = self->opt;

  node_t const& node = opt[nid];

  // above_nid is guaranteed to be a join node
  nid_t const& above_nid = node.ups[0];

  int n_inc_above = opt[above_nid].dims.size();

  std::vector<int> inc_above(n_inc_above);

  std::iota(inc_above.begin(), inc_above.end(), 0);

  vector<int> idxs_above = opt.get_out_for_input(inc_above, above_nid, nid);

  DCB_BEFORE_DYNAMIC("PREBLOCK LOCAL PART");
  Partition::pjoin_t& pabove = dynamic_cast<Partition::pjoin_t&>(*self->vars[above_nid]);

  IntVarArgs ret;
  for(auto const& idx: idxs_above) {
    ret << pabove.partition[idx];
  }
  return ret;
}

IntVarArgs
Partition::preblock_t::local_partition_below()
{
  auto const& opt = self->opt;

  node_t const& node = opt[nid];

  nid_t const& below_nid = opt.get_part_owner(node.downs[0]);

  int n_inc_below = opt[below_nid].dims.size();

  std::vector<int> inc_below(n_inc_below);

  std::iota(inc_below.begin(), inc_below.end(), 0);

  vector<int> idxs_below = opt.get_out(inc_below, below_nid);

  DCB_BEFORE_DYNAMIC("PREBLOCK LOCAL PART BELOW");
  IntVarArgs inc_part;
  if(self->opt[below_nid].type == node_t::node_type::input) {
    auto& p = dynamic_cast<Partition::pinput_t&>(*self->vars[below_nid]);
    inc_part << p.partition;
  } else {
    auto& p = dynamic_cast<Partition::pjoin_t&>(*self->vars[below_nid]);
    inc_part << p.partition;
  }

  IntVarArgs ret;
  for(auto const& idx: idxs_below) {
    ret << inc_part[idx];
  }
  return ret;
}

IntVarArgs
Partition::pagg_t::local_partition()
{
  auto const& opt = self->opt;

  node_t const& node = opt[nid];

  // the down node is guaraneeed to be a join node
  nid_t join_nid = node.downs[0];

  int n_inc = opt[join_nid].dims.size();

  std::vector<int> inc(n_inc);

  std::iota(inc.begin(), inc.end(), 0);

  vector<int> idxs = opt.get_out(inc, join_nid);

  DCB_BEFORE_DYNAMIC("PAGG LOCAL PART");
  Partition::pjoin_t& pjoin = dynamic_cast<Partition::pjoin_t&>(*self->vars[join_nid]);

  IntVarArgs ret;
  for(auto const& idx: idxs) {
    ret << pjoin.partition[idx];
  }
  return ret;
}

void Partition::preblock_t::propagate_is_no_op()
{
  IntVarArgs above = local_partition_above();
  IntVarArgs below = local_partition_below();

  assert(above.size() == below.size());
  BoolVarArgs args;
  for(int which = 0; which != below.size(); ++which) {
    BoolVar v = expr(*self, above[which] == below[which]);
    args << v;
  }

  // This node is a no op if and only if above part == below part
  rel(*self, BOT_AND, args, is_no_op);

  // If this is a no op, set a bunch of stuff to zero
  Partition::pode_t::_set_no_op(is_no_op);
}

void Partition::pode_t::_set_no_op(BoolVar& is_no_op) {
  // If this is a no op, the following are zero
  rel(*self, is_no_op >> (duration        == 0));
  rel(*self, is_no_op >> (worker          == 0));
  rel(*self, is_no_op >> (unit            == 0));
  rel(*self, is_no_op >> (kernel_duration == 0));
}

void Partition::pode_t::_set_unit_and_kernel_duration(int _unit, int _kernel_duration) {
  Int::IntView _kd = kernel_duration;
  Int::IntView _u  = unit;

  ModEvent _ev_kd = _kd.eq(*self, _unit            == 0 ? 0 : _kernel_duration);
  ModEvent _ev_u  = _u.eq (*self, _kernel_duration == 0 ? 0 : _unit);
  // ^ if one of these is zero, they are both zero

  // If one of these variables don't contain the required value in the domain,
  // the space should fail.

  if(_ev_kd == Int::ME_INT_FAILED ||
     _ev_u  == Int::ME_INT_FAILED)
  {
    self->fail();
  }
}

// guarantee that atleast one up node has the same blocking as this input node
void Partition::pinput_t::propagate_prefer_no_reblock() {
  node_t const& node = self->opt[nid];

  BoolVarArgs args;
  for(nid_t up: node.ups) {
    // It could be the case that the parent node is not yet covered!
    // (But it should never be the case that all up nodes of an input are not covered)
    if(self->vars[up] == nullptr) {
      continue;
    }

    node_t const& node_up = self->opt[up];
    if(node_up.type == node_t::node_type::reblock) {
      DCB_BEFORE_DYNAMIC("PROP INPUT PREFER REBLOCK");
      Partition::preblock_t& p_up =
        dynamic_cast<Partition::preblock_t&>(*self->vars[up]);
      args << p_up.is_no_op;
    } else
    if(node_up.type == node_t::node_type::join) {
      // It is guaranteed that all consecutive part owner nodes
      // have the same blocking. This happens via join's match_down_part_owners constraint.
      // This means the constraint on args does not need to be set.
      return;
    } else {
      assert(false);
    }
  }

  // atleast one of the up nodes must have a no op reblock
  rel(*self, BOT_OR, args, 1);
}

void Partition::pinput_t::propagate_no_cost_input() {
  rel(*self, start           == 0);
  rel(*self, duration        == 0);
  rel(*self, worker          == 0);
  rel(*self, kernel_duration == 0);
}

void Partition::pinput_t::when_partition_info_set() {
  nid_t _nid = nid;
  wait(*self, partition, [_nid](Space& home)
  {
    Partition& self = static_cast<Partition&>(home);
    Partition::pinput_t& p = static_cast<Partition::pinput_t&>(*self.vars[_nid]);

    int rank = p.rank();
    vector<int> inc_part;
    inc_part.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      inc_part.push_back(p.partition[i].val());
    }

    int unit = product(inc_part);
    int kernel_duration = self.opt.to_gecode_cost(
          self.opt.get_input_kernel_cost(inc_part, p.nid));
    DCB01("input nid "
      << _nid << " unit, kernel_duration: " << unit << ", " << kernel_duration);
    p._set_unit_and_kernel_duration(unit, kernel_duration);
  });
}

void Partition::pjoin_t::when_partition_info_set() {
  nid_t _nid = nid;
  wait(*self, partition, [_nid](Space& home)
  {
    Partition& self = static_cast<Partition&>(home);
    Partition::pjoin_t& p = static_cast<Partition::pjoin_t&>(*self.vars[_nid]);

    int rank = p.rank();
    vector<int> inc_part;
    inc_part.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      inc_part.push_back(p.partition[i].val());
    }

    int unit = product(inc_part);
    int kernel_duration = self.opt.to_gecode_cost(
          self.opt.get_join_kernel_cost(inc_part, p.nid));
    DCB01("join nid "
      << _nid << " unit, kernel_duration: " << unit << ", " << kernel_duration);
    p._set_unit_and_kernel_duration(unit, kernel_duration);
  });
}

// For each input that is also a partition owner, make sure we match indices
// with that guy
void Partition::pjoin_t::match_downs() {
  node_t const& node = self->opt[nid];

  for(int which = 0; which != node.downs.size(); ++which) {

    auto const& ordering = node.ordering[which];
    nid_t const& down_nid = node.downs[which];
    node_t const& down_node = self->opt[down_nid];

    IntVarArgs down_out;
    if(down_node.type == node_t::node_type::input) {
      down_out = static_cast<Partition::pinput_t&>(*self->vars[down_nid]).partition;
    } else
    if(down_node.type == node_t::node_type::join) {
      Partition::pjoin_t& down_p = static_cast<Partition::pjoin_t&>(*self->vars[down_nid]);

      std::vector<int> inc_idx(down_node.dims.size());
      std::iota(inc_idx.begin(), inc_idx.end(), 0);
      std::vector<int> idxs = self->opt.get_out(inc_idx, down_nid);

      for(int const& idx: idxs) {
        down_out << down_p.partition[idx];
      }
    } else
    if(down_node.type == node_t::node_type::agg) {
      // It is necessary this guys is the same shape as the input agg...
      Partition::pagg_t& down_p = static_cast<Partition::pagg_t&>(*self->vars[down_nid]);
      down_out = down_p.local_partition();
    } else
    if(down_node.type == node_t::node_type::reblock) {
      // There is nothing to do here since the reblock deduces it's partition
      // from this join.
      continue;
    } else {
      assert(false);
    }

    assert(down_out.size() == ordering.size());
    for(int i = 0; i != ordering.size(); ++i) {
      rel(*self, down_out[i] == partition[ordering[i]]);
    }
  }

}

void Partition::pagg_t::when_partition_info_set() {
  node_t const& node = self->opt[nid];

  nid_t _nid = nid;
  nid_t _join_nid = node.downs[0];

  Partition::pjoin_t& _pjoin = static_cast<Partition::pjoin_t&>(*self->vars[_join_nid]);
  IntVarArgs partition = _pjoin.partition;

  wait(*self, partition, [_nid, _join_nid](Space& home)
  {
    Partition& self = static_cast<Partition&>(home);
    Partition::pjoin_t& pjoin = static_cast<Partition::pjoin_t&>(*self.vars[_join_nid]);
    Partition::pagg_t&  pagg  = static_cast<Partition::pagg_t&> (*self.vars[_nid]);

    std::vector<int> inc_part;
    int rank = self.opt[_join_nid].dims.size();
    inc_part.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      inc_part.push_back(pjoin.partition[i].val());
    }

    int unit = product(self.opt.get_out(inc_part, _join_nid));
    int kernel_duration = self.opt.to_gecode_cost(
          self.opt.get_agg_kernel_cost(inc_part, pagg.nid));
    DCB01("agg nid "
      << _nid << " unit, kernel_duration: " << unit << ", " << kernel_duration);
    pagg._set_unit_and_kernel_duration(unit, kernel_duration);
  });
}

void Partition::preblock_t::when_partition_info_set() {
  node_t const& node = self->opt[nid];

  nid_t _nid = nid;

  IntVarArgs args_above = local_partition_above();
  IntVarArgs args_below = local_partition_below();
  assert(args_above.size() == args_below.size());

  IntVarArgs args = args_above + args_below;

  wait(*self, args, [_nid](Space& home)
  {
    Partition& self = static_cast<Partition&>(home);
    Partition::preblock_t& p = static_cast<Partition::preblock_t&>(*self.vars[_nid]);

    IntVarArgs args_above = p.local_partition_above();
    IntVarArgs args_below = p.local_partition_below();

    std::vector<int> above;
    std::vector<int> below;
    above.reserve(args_above.size());
    below.reserve(args_above.size());
    for(int i = 0; i != args_above.size(); ++i) {
      above.push_back(args_above[i].val());
      below.push_back(args_below[i].val());
    }

    int unit = product(above);
    int kernel_duration = self.opt.to_gecode_cost(
          self.opt.get_reblock_kernel_cost(above, below, p.nid));
    DCB01("reblock nid "
      << _nid << " unit, kernel_duration: " << unit << ", " << kernel_duration);
    p._set_unit_and_kernel_duration(unit, kernel_duration);
  });
}

void Partition::preblock_t::disallow_barrier_reblock() {
  // All this means is that no blocking along any dimension
  // can decrease. That guarantees that the reblock does not perform
  // as a barrier.
  IntVarArgs args_above = local_partition_above();
  IntVarArgs args_below = local_partition_below();
  for(int i = 0; i != args_above.size(); ++i) {
    rel(*self, args_above[i] >= args_below[i]);
  }
}

void Partition::_set_branching() {
  IntVarArgs starts;
  IntVarArgs workers;
  BoolVarArgs is_no_ops;
  IntVarArgs all_parts;
  for(nid_t const& nid: covered()) {
    node_t const& node = opt[nid];

    pode_t& p = *vars[nid];
    if(p.is_fixed()) {
      continue;
    }

    starts  << p.start;
    workers << p.worker;
    DCB_BEFORE_DYNAMIC("SET BRANCHING");
    if(node.type == node_t::node_type::agg) {
      pagg_t& pagg = dynamic_cast<pagg_t&>(p);
      is_no_ops << pagg.is_no_op;
    } else
    if(node.type == node_t::node_type::reblock) {
      preblock_t& preblock = dynamic_cast<preblock_t&>(p);
      is_no_ops << preblock.is_no_op;
    } else
    if(node.type == node_t::node_type::join) {
      pjoin_t& pjoin = dynamic_cast<pjoin_t&>(p);
      all_parts << pjoin.partition;
    } else
    if(node.type == node_t::node_type::input) {
      pinput_t& pinput = dynamic_cast<pinput_t&>(p);
      all_parts << pinput.partition;
    } else {
      assert(false);
    }
  }

  // TODO: how to branch?

  branch(*this, is_no_ops, BOOL_VAR_RND(rnd), BOOL_VAL_MAX());
  branch(*this, all_parts, INT_VAR_RND(rnd),  INT_VAL_RND(rnd));
  branch(*this, all_parts, INT_VAR_RND(rnd),  INT_VAL_MAX());
  branch(*this, workers,   INT_VAR_NONE(),    INT_VAL_RND(rnd));
  branch(*this, starts,    INT_VAR_RND(rnd),  INT_VAL_MIN());
  branch(*this, makespan,  INT_VAL_MIN());
}

vector<int> Partition::get_partition(nid_t nid) const {
  DCB_ACCESS_VAR("partition");
  nid_t owner_nid = opt.get_part_owner(nid);
  node_t const& owner_node = opt[owner_nid];

  vector<int> inc_part;
  inc_part.reserve(owner_node.dims.size());

  IntVarArgs partition;
  if(owner_node.type == node_t::node_type::input) {
    DCB_BEFORE_DYNAMIC("GET PART");
    partition = dynamic_cast<pinput_t&>(*(vars[owner_nid])).partition;
  } else
  if(owner_node.type == node_t::node_type::join) {
    DCB_BEFORE_DYNAMIC("GET PART");
    partition = dynamic_cast<pjoin_t&>(*(vars[owner_nid])).partition;
  } else {
    assert(false);
  }

  for(int i = 0; i != owner_node.dims.size(); ++i) {
    inc_part.push_back(partition[i].val());
  }

  return opt.get_node_incident(inc_part, nid);
}

int Partition::get_start(nid_t nid)     const {
  DCB_ACCESS_VAR("start");
  return vars[nid]->start.val();
}

int Partition::get_duration(nid_t nid)  const {
  DCB_ACCESS_VAR("duration");
  return vars[nid]->duration.val();
}

int Partition::get_worker(nid_t nid)    const {
  DCB_ACCESS_VAR("worker");
  return vars[nid]->worker.val();
}

int Partition::get_unit(nid_t nid)      const {
  DCB_ACCESS_VAR("unit");
  return vars[nid]->unit.val();
}

//#define _FAIL_IF_BAD(x) { \
//  ModEvent ret = x;     \
//  if(ret == Int::ME_INT_FAILED) { self->fail(); } \
//}
#define _FAIL_IF_BAD(x) { \
  ModEvent ret = x;     \
  if(ret == Int::ME_INT_FAILED) { throw std::runtime_error("BWOAH"); } \
}

void Partition::pode_t::fix(pode_t const& other) {
  Int::IntView _start(            start);
  Int::IntView _end(              end);
  Int::IntView _duration(         duration);
  Int::IntView _worker(           worker);
  Int::IntView _unit(             unit);
  Int::IntView _kernel_duration(  kernel_duration);

  // Note that start and end are min, since there can be sliding
  // windows of possibilities where the overall makespan doesn't matter.
  // But the rest there must be exactly one variable in the domain

  _FAIL_IF_BAD(_start.eq(            *self, other.start.min()));
  _FAIL_IF_BAD(_end.eq(              *self, other.end.min()));

  _FAIL_IF_BAD(_duration.eq(         *self, other.duration.val()));
  _FAIL_IF_BAD(_worker.eq(           *self, other.worker.val()));
  _FAIL_IF_BAD(_unit.eq(             *self, other.unit.val()));
  _FAIL_IF_BAD(_kernel_duration.eq(  *self, other.kernel_duration.val()));

  fixed = true;
}

void Partition::pagg_t::fix(pode_t const& other) {
  Partition::pode_t::fix(other);

}
void Partition::preblock_t::fix(pode_t const& other) {
  Partition::pode_t::fix(other);

}
void Partition::pjoin_t::fix(pode_t const& other) {
  Partition::pode_t::fix(other);

  pjoin_t other_input = static_cast<Partition::pjoin_t const&>(other);

  for(int i = 0; i != partition.size(); ++i) {
    Int::IntView _p(partition[i]);
    _FAIL_IF_BAD(_p.eq(*self, other_input.partition[i].val()));
  }
}
void Partition::pinput_t::fix(pode_t const& other) {
  Partition::pode_t::fix(other);

  pinput_t other_input = static_cast<Partition::pinput_t const&>(other);

  for(int i = 0; i != partition.size(); ++i) {
    Int::IntView _p(partition[i]);
    _FAIL_IF_BAD(_p.eq(*self, other_input.partition[i].val()));
  }
}

// Some rules:
//   0. All dependent nodes are included
//   1. If an input is included, then so is atleast one of it's ups
//   2. If a reblock is included, so is the parent join
//   3. If a join is included, so are parent joins and parent aggs
// Assumption:
//   input so_far always has this constraint
void _add(
  vector<node_t> const& dag,
  std::set<nid_t>& so_far,
  int& num_added,
  nid_t nid)
{
  node_t const& node = dag[nid];

  // (1) is satisfied by assuming this is only called when
  // adding other functions
  if(node.type == node_t::node_type::input) {
    auto [_0, did_insert] = so_far.insert(nid);
    if(did_insert) {
      num_added++;

      bool has_an_included_parent = false;
      for(nid_t const& up: node.ups) {
        if(so_far.count(up) > 0) {
          has_an_included_parent = true;
          break;
        }
      }
      if(!has_an_included_parent) {
        throw std::runtime_error("_add incorrectly added an input node");
      }
    }
    return;
  }

  auto [_0, did_insert] = so_far.insert(nid);
  if(did_insert) {
    num_added++;
  } else {
    // If this nid has been added before, either it's dependencies
    // have been added or are being added
    return;
  }

  // add all downs (0)
  for(nid_t const& down: node.downs) {
    _add(dag, so_far, num_added, down);
  }

  // does rule (2) apply?
  if(node.type == node_t::node_type::reblock) {
    return _add(dag, so_far, num_added, node.ups[0]);
  }

  // does rule (3) apply?
  if(node.type == node_t::node_type::join) {
    for(nid_t const& up: node.ups) {
      // a join up is either a reblock, a join or an agg.
      if(dag[up].type != node_t::node_type::reblock) {
        _add(dag, so_far, num_added, up);
      }
    }
  }
}

// TODO: What is the best way to set the covering?
// It is tricky because you can end up with unsolveable
// solutions, depending on what parameters were chosen

void Partition::_set_covering(vector<nid_t> const& must_cover_these) {
  std::set<nid_t> so_far;
  so_far.insert(must_cover_these.begin(), must_cover_these.end());

  int num_added = 0;

  auto const& dag = opt.get_dag();

  vector<nid_t> const& ordering = opt.breadth_dag_order();

  for(nid_t const& nid: ordering) {
    node_t const& node = opt[nid];

    if(num_added >= opt.cover_size()) {
      break;
    }

    if(node.type == node_t::node_type::input) {
      continue;
    }

    _add(dag, so_far, num_added, nid);
  }

  // copy to _covered
  _covered.resize(so_far.size());
  std::copy(so_far.begin(), so_far.end(), _covered.begin());
}

}}

#include "partition.h"

#include "../misc.h"

#include <unordered_map>
#include <numeric>
#include <set>

#define DCB_BEFORE_DYNAMIC(x) //std::cout << __LINE__ << " " << x << std::endl
#define DCB_ACCESS_VAR(x)     //std::cout << __LINE__ << " " << x << std::endl;
#define DCB_P_CONSTRUCTOR(x)  //std::cout << __LINE__ << " " << x << std::endl;
#define DCB01(x)              //std::cout << __LINE__ << " " << x << std::endl;
#define DCB_WHEN(x)           //std::cout << __LINE__ << " " << x << std::endl;
#define DCB02(x)              //std::cout << x << std::endl;
#define DCB_COVER(x)          //std::cout << __LINE__ << " " << x << std::endl;
#define DCB03(x)              //std::cout << __LINE__ << " " << x << std::endl

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
              60,64,72,84,96,108,120,132,144,256})
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
        std::cout << p << " ";
      }
    }
    std::cout << std::endl;
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
  Partition* self_, nid_t nid_, int lower_limit, int upper_limit):
    self(self_), nid(nid_)
{
  start    = IntVar(*self, lower_limit, upper_limit);
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
    if(self->vars[nid]) {
      // We have the child node, use an `IntVar`
      pode_t& child = *(self->vars[nid]);
      rel(*self, child.end <= start, self->opt.ipl());
    } else {
      // there is nothing to be done since this node will already start
      // after all the fixed nodes.. But we could do this:
      //   partition_info_t const& child = partition_info[nid];
      //   int end = child.start + child.duration;
      //   rel(*self, end <= start, self->opt.ipl());
    }
  }

  // the makespan is done after this node is done
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
  Partition* self, nid_t nid, int lower_limit, int upper_limit):
    Partition::pode_t(self, nid, lower_limit, upper_limit)
{
  is_no_op = BoolVar(*self, 0, 1);
}

Partition::preblock_t::preblock_t(
  Partition* self, nid_t nid, int lower_limit, int upper_limit):
    Partition::pode_t(self, nid, lower_limit, upper_limit)
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
  Partition* self, nid_t nid, int lower_limit, int upper_limit):
    Partition::pode_t(self, nid, 0, upper_limit)
{
  CONSTRUCTOR_PARTITION
}

Partition::pjoin_t::pjoin_t(
  Partition* self, nid_t nid, int lower_limit, int upper_limit):
    Partition::pode_t(self, nid, lower_limit, upper_limit)
{
  CONSTRUCTOR_PARTITION
}

void Partition::_set_cumulative() {
  IntVarArgs start;
  IntVarArgs duration;
  IntVarArgs end;
  IntVarArgs worker;
  for(nid_t const& nid: covering()) {
    pode_t& p = *vars[nid];
    start    << p.start;
    duration << p.duration;
    end      << p.end;
    worker   << p.worker;
  }

  _cumulative(opt.num_workers(), start, duration, end, worker);
}

Partition::Partition(
  partition_options_t const& opt0,
  vector<partition_info_t> const& info0):
    IntMinimizeSpace(),
    opt(opt0),
    info(info0),
    rnd(opt0.seed())
{
  DCB_P_CONSTRUCTOR("A");

  _set_covering();

  DCB_P_CONSTRUCTOR("B");

  int lower_limit = 0;
  for(auto const& p: info) {
    if(p.is_set()) {
      lower_limit = std::max(lower_limit, p.start + p.duration);
    }
  }
  int upper_limit = 100 * opt.upper_bound_time();

  makespan = IntVar(*this, lower_limit, upper_limit);

  DCB_P_CONSTRUCTOR("C");

  vars = vector<pode_ptr>(opt.size());
  for(auto& ptr: vars) {
    ptr = nullptr;
  }

  for(nid_t const& nid: covering()) {
    node_t const& node = opt[nid];
    if(node.type == node_t::node_type::input) {
      vars[nid] = std::unique_ptr<pode_t>(new pinput_t(this, nid, lower_limit, upper_limit));
    } else
    if(node.type == node_t::node_type::join) {
      vars[nid] = std::unique_ptr<pode_t>(new pjoin_t(this, nid, lower_limit, upper_limit));
    } else
    if(node.type == node_t::node_type::agg) {
      vars[nid] = std::unique_ptr<pode_t>(new pagg_t(this, nid, lower_limit, upper_limit));
    } else
    if(node.type == node_t::node_type::reblock) {
      vars[nid] = std::unique_ptr<pode_t>(new preblock_t(this, nid, lower_limit, upper_limit));
    } else {
      throw std::runtime_error("::Partition should not reach");
    }
  }
  DCB_P_CONSTRUCTOR("D");

  // Set the dag ordering and the makespan and wtvr else
  // the base constraint does
  for(nid_t const& nid: covering()) {
    vars[nid]->set_base_constraint();
  }

  DCB_P_CONSTRUCTOR("E");

  // For each node, set the partition related constraints specific to that node
  for(nid_t const& nid: covering()) {
    //std::cout << nid << ", ";
    //node_t const& node = opt[nid];
    //if(node.type == node_t::node_type::input) {
    //  std::cout << "input";
    //} else
    //if(node.type == node_t::node_type::join) {
    //  std::cout << "join";
    //} else
    //if(node.type == node_t::node_type::agg) {
    //  std::cout << "agg";
    //} else
    //if(node.type == node_t::node_type::reblock) {
    //  std::cout << "reblock";
    //}
    //std::cout << std::endl;
    vars[nid]->set_constraints();
  }

  DCB_P_CONSTRUCTOR("F");

  _set_cumulative();

  DCB_P_CONSTRUCTOR("G");

  _set_branching();

  DCB_P_CONSTRUCTOR("H");
}

Partition::Partition(Partition& other):
  IntMinimizeSpace(other),
  opt(other.opt),
  info(other.info),
  rnd(other.rnd),
  _covering(other._covering),
  _num_covered(other._num_covered)
{
  DCB_COVER("ENTERED Partition::Partition(Partition& other)");

  makespan.update(*this, other.makespan);

  vars = vector<pode_ptr>(opt.size());
  for(auto& ptr: vars) {
    ptr = nullptr;
  }

  // static cast should also be valid, right?
  for(nid_t const& nid: covering()) {
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
  nid(other.nid)
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
  for(nid_t const& nid: covering()) {
    auto& p = *vars[nid];
    total += p.worker.min() * p.duration.min();
  }
  for(auto const& p: info) {
    if(p.is_set()) {
      total += p.worker * p.duration;
    }
  }
  //std::cout << "total, num workers, makespan: " << total << ", " << opt.num_workers() << ", " << makespan << std::endl;
  float utilization = 100.0 * (total / (1.0*opt.num_workers()*makespan.max()));

  os << "Partition makespan, utilization:  " << makespan.max() << ",  ";
  os << utilization << "%" << std::endl;
//  os << "Partition makespan: " << makespan.max() << std::endl;
}

void Partition::pagg_t::propagate_is_no_op()
{
  DCB03("agg propage is no op");
  if(nid == 323) {
    DCB03("turning off agg nid 323 propaget");
    return;
  }

  auto const& opt = self->opt;
  node_t const& node = opt[nid];

  nid_t join_nid = node.downs[0];
  node_t const& join_node = opt[join_nid];

  BoolVarArgs args;
  if(self->vars[join_nid]) {
    DCB_BEFORE_DYNAMIC("PAGG PROP");
    Partition::pjoin_t& pjoin = dynamic_cast<Partition::pjoin_t&>(*self->vars[join_nid]);

    for(auto const& which: join_node.aggs) {
      BoolVar v = expr(*self, pjoin.partition[which] == 1);
      args << v;
    }
  } else {
    throw std::runtime_error(
      "The agg's corresponding join is not included in this cover");
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
  if(!self->vars[above_nid]) {
    throw std::runtime_error("reblocks up node not included in this cover");
  }
  Partition::pjoin_t& pabove = dynamic_cast<Partition::pjoin_t&>(*self->vars[above_nid]);

  IntVarArgs ret;
  for(auto const& idx: idxs_above) {
    ret << pabove.partition[idx];
  }
  return ret;
}

bool Partition::preblock_t::is_below_fixed() const {
  nid_t const& below_nid = self->opt[nid].downs[0];
  return self->info[below_nid].is_set();
}

vector<int>
Partition::preblock_t::local_fixed_partition_below()
{
  nid_t const& below_nid = self->opt[nid].downs[0];
  if(!self->info[below_nid].is_set()) {
    throw std::runtime_error(
      "reblocks below node is not fixed; call local_partition_below instead");
  }

  auto const& inc_below = self->info[below_nid].blocking;
  return self->opt.get_out(inc_below, below_nid);
}

IntVarArgs
Partition::preblock_t::local_partition_below()
{
  auto const& opt = self->opt;

  node_t const& node = opt[nid];

  nid_t const& below_nid = opt.get_part_owner(node.downs[0]);
  if(!self->vars[below_nid]) {
    throw std::runtime_error(
      "reblocks below node is fixed; call local_fixed_partition_below instad");
  }

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
  if(!self->vars[join_nid]) {
    throw std::runtime_error(
      "in pagg_t::local_partition, vars is not covered");
  }

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
  node_t const& node = self->opt[nid];
  nid_t below_nid = node.downs[0];

  BoolVarArgs args;
  if(self->info[below_nid].is_set()) {
    DCB03("reblock's propagate is no op");

    if(nid == 320 || nid == 321) {
      DCB03("turning off reblock's propagate");
      return;
    }

    IntVarArgs above = local_partition_above();
    vector<int> below = local_fixed_partition_below();

    assert(above.size() == below.size());
    for(int which = 0; which != below.size(); ++which) {
      BoolVar v = expr(*self, above[which] == below[which]);
      args << v;
    }
  } else {
    DCB03("not gonna happen");

    IntVarArgs above = local_partition_above();
    IntVarArgs below = local_partition_below();

    assert(above.size() == below.size());
    for(int which = 0; which != below.size(); ++which) {
      BoolVar v = expr(*self, above[which] == below[which]);
      args << v;
    }
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
  rel(*self, unit            == 0);
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
    DCB_WHEN("input nid "
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
    DCB_WHEN("join nid "
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
    if(!self->vars[down_nid]) {
      DCB03("not gonna happen");
      // In this case, we know down out...
      auto down_out_fixed = self->opt.get_out(
        self->info[down_nid].blocking,
        down_nid);
      for(int const& val: down_out_fixed) {
        down_out << IntVar(*self, val, val);
      }
    } else if(down_node.type == node_t::node_type::input) {
        DCB03("not gonna happen");
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
      DCB03("matching down reblock " << down_nid << ",  nothing to do ");
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
    DCB_WHEN("agg nid "
      << _nid << " unit, kernel_duration: " << unit << ", " << kernel_duration);
    pagg._set_unit_and_kernel_duration(unit, kernel_duration);
  });
}

void Partition::preblock_t::when_partition_info_set() {
  node_t const& node = self->opt[nid];

  nid_t _nid = nid;

  IntVarArgs args;
  IntVarArgs args_above = local_partition_above();
  if(is_below_fixed()) {
    args = args_above;
  } else {
    IntVarArgs args_below = local_partition_below();
    assert(args_above.size() == args_below.size());

    args = args_above + args_below;
  }

  wait(*self, args, [_nid](Space& home)
  {
    Partition& self = static_cast<Partition&>(home);
    Partition::preblock_t& p = static_cast<Partition::preblock_t&>(*self.vars[_nid]);

    std::vector<int> above;
    std::vector<int> below;

    IntVarArgs args_above = p.local_partition_above();
    above.reserve(args_above.size());
    for(int i = 0; i != args_above.size(); ++i) {
      above.push_back(args_above[i].val());
    }

    if(p.is_below_fixed()) {
      below = p.local_fixed_partition_below();
    } else {
      IntVarArgs args_below = p.local_partition_below();
      below.reserve(args_below.size());
      for(int i = 0; i != args_below.size(); ++i) {
        below.push_back(args_below[i].val());
      }
    }

    // If for all i, above[i] >= below[i] then this reblock does not induce a barrier
    // Otherwise it is called a barrier reblock, even though it might not be that bad of
    // a barrier, i.e. [3,4,5,5] -> [1,1,5,5] is not that bad of a barrier, but
    // [4,1] -> [1,3] is...
    bool is_barrier_reblock = false;
    for(int i = 0; i != above.size(); ++i) {
      if(above[i] < below[i]) {
        is_barrier_reblock = true;
        break;
      }
    }

    int unit = product(above);
    int kernel_duration = self.opt.to_gecode_cost(
          self.opt.get_reblock_kernel_cost(above, below, p.nid));
    if(kernel_duration > 0 && is_barrier_reblock) {
      kernel_duration += self.opt.barrier_reblock_cost();
    }
    DCB_WHEN("reblock nid "
      << _nid << " unit, kernel_duration: " << unit << ", " << kernel_duration);
    p._set_unit_and_kernel_duration(unit, kernel_duration);
  });
}

void Partition::_set_branching() {
  IntVarArgs starts;
  IntVarArgs workers;
  BoolVarArgs is_no_ops;
  IntVarArgs all_parts;
  for(nid_t const& nid: covering()) {
    DCB_WHEN("THIS GUYS COVERED ATM: " << nid);
    node_t const& node = opt[nid];

    pode_t& p = *vars[nid];

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

  // TODO(optimize): how to branch?

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

  if(!vars[owner_nid]) {
    throw std::runtime_error("invalid call to Partition::get_partition");
  }

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

void Partition::pode_t::print_domain_sizes() {
  std::cout << "start:          " << start.size()           << std::endl;
  std::cout << "end:            " << end.size()             << std::endl;
  std::cout << "duration        " << duration.size()        << std::endl;
  std::cout << "worker          " << worker.size()          << std::endl;
  std::cout << "unit            " << unit.size()            << std::endl;
  std::cout << "kernel_duration " << kernel_duration.size() << std::endl;
}
void Partition::pagg_t::print_domain_sizes() {
  std::cout << "[[----------------------------------\n";
  Partition::pode_t::print_domain_sizes();
  std::cout << "----------------------------------]]\n";
}
void Partition::preblock_t::print_domain_sizes() {
  std::cout << "[[----------------------------------\n";
  Partition::pode_t::print_domain_sizes();
  std::cout << "----------------------------------]]\n";
}
void Partition::pjoin_t::print_domain_sizes() {
  std::cout << "[[----------------------------------\n";
  Partition::pode_t::print_domain_sizes();
  for(int i = 0; i != partition.size(); ++i) {
    std::cout << "partition[i]" << partition[i].size() << std::endl;
  }
  std::cout << "----------------------------------]]\n";
}
void Partition::pinput_t::print_domain_sizes() {
  std::cout << "[[----------------------------------\n";
  Partition::pode_t::print_domain_sizes();
  for(int i = 0; i != partition.size(); ++i) {
    std::cout << "partition[i]" << partition[i].size() << std::endl;
  }
  std::cout << "----------------------------------]]\n";
}

void _add(
  vector<node_t> const& dag,
  std::set<nid_t>& so_far,
  int& num_added,
  nid_t nid)
{
  auto [_0, did_insert] = so_far.insert(nid);
  if(did_insert) {
    num_added++;
  } else {
    // If this nid has been added before, either it's dependencies
    // have been added or are being added
    return;
  }

  node_t const& node = dag[nid];

  for(nid_t const& down: node.downs) {
    _add(dag, so_far, num_added, down);
  }
}

void Partition::_set_covering() {
  std::set<nid_t> so_far;
  for(int nid = 0; nid != opt.size(); ++nid) {
    auto const& p = info[nid];
    if(p.is_set()) {
      so_far.insert(nid);
    }
  }

  int num_added = 0;

  auto const& dag = opt.get_dag();

  vector<nid_t> const& ordering = opt.breadth_dag_order();

  for(nid_t const& nid: ordering) {
    node_t const& node = opt[nid];

    if(num_added >= opt.cover_size()) {
      break;
    }

    // only add join nodes, unless the join has an associated agg, then add the agg.
    if(node.type == node_t::node_type::join) {
      if(node.ups.size() == 1 && opt[node.ups[0]].type == node_t::node_type::agg) {
        auto const& agg_nid = node.ups[0];
        _add(dag, so_far, num_added, agg_nid);
      } else {
        _add(dag, so_far, num_added, nid);
      }
    }
  }

  // set _num_covered
  _num_covered = so_far.size();

  // copy to _covering
  _covering.reserve(opt.cover_size()*2);
  for(auto const& nid: so_far) {
    if(!info[nid].is_set()) {
      _covering.push_back(nid);
    }
  }
}


}}

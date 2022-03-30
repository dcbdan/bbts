#include "partition.h"
#include <set>

namespace bbts { namespace dag {

using namespace Gecode;

//#define DCB1(x) std::cout << x << std::endl
#define DCB1(x)
//#define DCB2(x) std::cout << x << std::endl;
#define DCB2(x)

using Gecode::Int::IntView;

partition_init_t Partition::build_init(partition_options_t const& opt) {
  vector<node_t> const& dag = opt.get_dag();

  partition_init_t ret(dag.size());

  {
    // For a partitioning where all join nodes are maximally partitioned,
    // determine how long each node takes
    vector<vector<dim_t>> partitionings = opt.get_max_full_dag_partitioning();
    vector<int> durations;
    for(nid_t nid = 0; nid != dag.size(); ++nid) {
      auto [d, _] = opt.get_duration_worker_pair(partitionings, nid);
      durations.push_back(d);
    }

    // Use that info to determine the max possible time
    ret.max_possible_time = 1; // just in case
    for(auto t: durations) {
      ret.max_possible_time += t;
    }

    // Take the duration variable and set every node except join nodes
    // to have zero computation. Use that to compute a lower bound
    // for the start times of all nodes
    for(nid_t nid = 0; nid != dag.size(); ++nid) {
      node_t const& node = dag[nid];
      if(node.type != node_t::node_type::join) {
        durations[nid] = 0;
      }
    }
    auto min_times_complete = opt.time_to_completion(durations);

    for(int i = 0; i != dag.size(); ++i) {
      int min_time_start = min_times_complete[i] - durations[i];
      ret.start_min[i] = min_time_start;
      ret.start_max[i] = ret.max_possible_time;
    }

    // min possible time really means lower bound
    ret.min_possible_time = 0;
    for(auto v: min_times_complete) {
      ret.min_possible_time = std::max(ret.min_possible_time, v);
    }
  }

  // No branching will happen on duration, end or worker nodes.
  // And duration, worker pairs will be set when the corresponding partition
  // information is known.
  // So don't worry about restricting their domains.
  // But I'm not sure for zerod nodes what cumulative enforces..
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    if(dag[nid].type == node_t::node_type::input) {
      ret.duration_min[nid] = 0;
      ret.duration_max[nid] = 0;
      ret.worker_min[nid] = 0;
      ret.worker_max[nid] = 0;
    } else {
      ret.duration_min[nid] = 0;
      ret.duration_max[nid] = ret.max_possible_time;
      ret.worker_min[nid] = 0;
      ret.worker_max[nid] = opt.get_num_workers();
    }
  }

  // iniitialize partitions
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.type == node_t::node_type::join ||
       node.type == node_t::node_type::input)
    {
      ret.partitions_min[nid] = 0;
      ret.partitions_max[nid] = opt.all_partitions(nid).size() - 1;
      // partitions[nid] is an index into opt.all_partitions(nid)
    } else {
      // this nid partitions isn't gonna be used
      ret.partitions_min[nid] = 0;
      ret.partitions_max[nid] = 0;
    }
  }
  return ret;
}

Partition::Partition(Partition const& other, partition_init_t const& info):
  IntMinimizeSpace(),
  opt(other.opt),
  rnd(other.rnd),
  cover_to(other.cover_to + opt.cover_size())
{
  vector<node_t> const& dag = opt.get_dag();

  makespan = IntVar(*this, 0, info.max_possible_time);

  start      = IntVarArray(*this, dag.size());
  duration   = IntVarArray(*this, dag.size());
  end        = IntVarArray(*this, dag.size());
  worker     = IntVarArray(*this, dag.size());
  partitions = IntVarArray(*this, dag.size());

  auto _prev_nids = _get_nids(other.cover_to);
  std::set<nid_t> prev_nids(_prev_nids.begin(), _prev_nids.end());
  auto was_covered = [&prev_nids](nid_t nid) {
    return prev_nids.count(nid) == 1;
  };
  auto same_domain_var = [&](IntVar const& var){
    IntVarRanges rngs(var);
    return IntVar(*this, IntSet(rngs));
  };

  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    if(was_covered(nid)) {
      //std::cout << nid << " " << dag[nid].type << ": (" << other.start[nid] << ", " << other.end[nid] << ") @ " << other.worker[nid] << " | " << other.partitions[nid] << "                   " << dag[nid] << std::endl;
      start[nid]      = same_domain_var(other.start[nid]);
      duration[nid]   = same_domain_var(other.duration[nid]);
      end[nid]        = same_domain_var(other.end[nid]);
      worker[nid]     = same_domain_var(other.worker[nid]);
      partitions[nid] = same_domain_var(other.partitions[nid]);
    } else {
      //std::cout << nid << std::endl;
      start[nid]      = info.start(     *this, nid);
      duration[nid]   = info.duration(  *this, nid);
      end[nid]        = info.end(       *this, nid);
      worker[nid]     = info.worker(    *this, nid);
      partitions[nid] = info.partitions(*this, nid);
    }
  }

  _init_propagations();

  int n = _cover_propagations_and_branches(cover_to);

  std::cout << "COPYING PREVIOUSLY COVERED SPACE" << std::endl;
  std::cout << "LOWER_BOUND_TIME: " << info.min_possible_time << std::endl;
  std::cout << "UPPER_BOUND_TIME: " << info.max_possible_time << std::endl;
  std::cout << "DAG SIZE: " << dag.size() << std::endl;
  std::cout << "COVER TO: " << n << std::endl;
}

Partition::Partition(partition_options_t const& opt0, partition_init_t const& info):
  IntMinimizeSpace(),
  opt(opt0),
  rnd(opt0.seed()),
  cover_to(opt.cover() ? opt.cover_size() : opt.get_dag().size())
{
  vector<node_t> const& dag = opt.get_dag();

  makespan = IntVar(*this, 0, info.max_possible_time);

  start      = IntVarArray(*this, dag.size());
  duration   = IntVarArray(*this, dag.size());
  end        = IntVarArray(*this, dag.size());
  worker     = IntVarArray(*this, dag.size());
  partitions = IntVarArray(*this, dag.size());

  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    start[nid]      = info.start(     *this, nid);
    duration[nid]   = info.duration(  *this, nid);
    end[nid]        = info.end(       *this, nid);
    worker[nid]     = info.worker(    *this, nid);
    partitions[nid] = info.partitions(*this, nid);
  }

  _init_propagations();

  int n = _cover_propagations_and_branches(cover_to);

  std::cout << "LOWER_BOUND_TIME: " << info.min_possible_time << std::endl;
  std::cout << "UPPER_BOUND_TIME: " << info.max_possible_time << std::endl;
  std::cout << "DAG SIZE: " << dag.size() << std::endl;
  std::cout << "COVER TO: " << n << std::endl;
}

void Partition::_init_propagations() {
  vector<node_t> const& dag = opt.get_dag();

  // set end directly
  end = IntVarArray(*this, dag.size());
  for(nid_t i = 0; i != dag.size(); ++i) {
    end[i] = expr(*this, start[i] + duration[i]);
  }

  // Now whenever a partition takes on a single value, assign the
  // join and agg nodes duration and worker values
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.type == node_t::node_type::join) {
      wait(*this, partitions[nid],
        [nid](Space& home) {
          Partition* self = static_cast<Partition*>(&home);
          return self->assign_join_agg(nid);
       });
    }
  }

  // Now whenever a reblock from above and below is assigned, set the duration and worker
  // for that node
  for(node_t const& node: dag) {
    if(node.type == node_t::node_type::reblock) {
      nid_t nid = node.id;
      nid_t join_above          = node.ups[0];
      nid_t join_or_input_below = opt.get_compute_nid(node.downs[0]);
      IntVarArgs corresponding_parts({
        partitions[join_above],
        partitions[join_or_input_below]});
      wait(*this, corresponding_parts,
        [nid](Space& home) {
          Partition* self = static_cast<Partition*>(&home);
          return self->assign_reblock(nid);
       });
    }
  }
}

int Partition::_cover_propagations_and_branches(int num_compute_cover) {
  //  < propagators >
  //  - set dag ordering
  //  - set cumulative constraint
  //  - set make span constraint
  //  < branch >
  //  - set branch partitions
  //  - set branch start
  //  - set branch makespan

  vector<node_t> const& dag = opt.get_dag();

  vector<nid_t> nids = _get_nids(num_compute_cover);

  // traverse the dag and schedule ordering constraints:
  //   each node can start after the inputs have completed
  for(nid_t nid: nids) {
    for(nid_t prev_nid: dag[nid].downs) {
      // start prev: 0
      // duration prev: 3
      // [0,3) <= [3,duration)
      rel(*this,
          end[prev_nid] <= start[nid],
          opt.ipl());
    }
  }

  // We only care about the subset we are covering,
  // so get args for that subset.
  IntVarArgs _start(nids.size());
  IntVarArgs _duration(nids.size());
  IntVarArgs _end(nids.size());
  IntVarArgs _worker(nids.size());
  IntVarArgs _partitions(nids.size());
  for(int which = 0; which != nids.size(); ++which) {
    nid_t nid = nids[which];
    _start[which]      = start[nid];
    _duration[which]   = duration[nid];
    _end[which]        = end[nid];
    _worker[which]     = worker[nid];
    _partitions[which] = partitions[nid];
  }

  // make sure worker capacity is not violated
  _cumulative(
    opt.get_num_workers(),
    _start,
    _duration,
    _end,
    _worker,
    opt.ipl());

  // set the makespan
  for(nid_t nid: nids) {
    rel(*this, end[nid] <= makespan, opt.ipl());
  }

  // Now we have the propagators set, but we need to do the branching.
  for(int i = 0; i != dag.size(); ++i) {
    branch(*this, _partitions, INT_VAR_NONE(), INT_VAL_RND(rnd));
  }
  branch(*this, _start, INT_VAR_RND(rnd), INT_VAL_MIN());
  branch(*this, makespan, INT_VAL_MIN());

  return nids.size();
}

void Partition::assign_join_agg(nid_t join_nid) {
  DCB2("assign join agg");
  // assumption: the join has one partition in it's domain
  vector<int> inc_part = opt.get_which_partition(join_nid, partitions[join_nid].val());

  {
    auto [d,w] = opt.get_est_duration_worker_pair(inc_part, join_nid);
    IntView _d = duration[join_nid];
    IntView _w = worker[join_nid];
    if(_d.in(d) && _w.in(w)) {
      _d.eq(*this, d);
      _w.eq(*this, w);
    } else {
      this->fail();
      DCB2("---------------");
      return;
    }
  }

  {
    nid_t agg_nid = opt.get_dag()[join_nid].ups[0];
    auto [d,w] = opt.get_est_duration_worker_pair(inc_part, agg_nid);
    IntView _d = duration[agg_nid];
    IntView _w = worker[agg_nid];
    if(_d.in(d) && _w.in(w)) {
      _d.eq(*this, d);
      _w.eq(*this, w);
    } else {
      this->fail();
      DCB2("---------------");
      return;
    }
  }
  DCB2("---------------");
}

void Partition::assign_reblock(nid_t reblock_nid) {
  DCB2("ASSIGN REBLOCK " << std::boolalpha << failed());
  node_t const& reblock_node = opt.get_dag()[reblock_nid];

  nid_t nid_above = reblock_node.ups[0];
  nid_t nid_below = opt.get_compute_nid(reblock_node.downs[0]);

  vector<int> inc_above = opt.get_which_partition(nid_above, partitions[nid_above].val());
  vector<int> inc_below = opt.get_which_partition(nid_below, partitions[nid_below].val());

  IntView _d = duration[reblock_nid];
  IntView _w = worker[reblock_nid];

  DCB2("setting d and w");
  auto above = opt.get_reblock_out(inc_above, reblock_nid);
  auto below = opt.get_out(inc_below, nid_below);
  if(above == below) {
    if(_d.in(0) && _w.in(0)) {
      _d.eq(*this, 0);
      _w.eq(*this, 0);
    } else {
      this->fail();
    }
  } else {
    auto [d,w] = opt.get_est_duration_worker_pair(inc_above, reblock_nid);
    if(_d.in(d) && _w.in(w)) {
      _d.eq(*this, d);
      _w.eq(*this, w);
    } else {
      this->fail();
    }
  }
  DCB2("--------------");
}

int Partition::select_partition(nid_t nid) const {
  vector<node_t> const& dag = opt.get_dag();
  node_t const& node = dag[nid];

  IntVar const& domain = partitions[nid];
  vector<vector<dim_t>> const& parts = opt.all_partitions(nid);

  int num_select_branch = 10;
  // For the first num_select_branch partitions, pick the one with
  // the least score..Here, a score is +1 for an agg, +1 for each reblock.
  int cnt = 0;
  int bst = 100000;
  int ret = 0;
  for(IntVarValues i(domain); i(); ++i) {
    if(cnt == num_select_branch)
      break;
    /////////////
    vector<dim_t> const& inc_part = parts[i.val()];
    int score = 0;

    // Will there be an agg?
    for(int const& a: node.aggs) {
      if(inc_part[a] > 1) {
        score++;
        break;
      }
    }

    //// How many reblocks will there be?
    for(nid_t reblock_nid: node.downs) {
      node_t const& reblock = dag[reblock_nid];
      nid_t down_compute = opt.get_compute_nid(reblock.downs[0]);

      // If the down node isn't set, don't score it
      if(partitions[down_compute].size() == 1) {
        vector<dim_t> const& inc_down = opt.get_which_partition(
          down_compute,
          partitions[down_compute].val());
        auto out_down = opt.get_out(inc_down, down_compute);
        auto out_part = opt.get_reblock_out(inc_part, reblock_nid);
        if(out_down != out_part) {
          score++;
        }
      }
    }

    if(score < bst) {
      ret = i.val();
      bst = score;
    }
    /////////////
    cnt++;
  }
  return ret;
}

void Partition::print(std::ostream& os) const {
  vector<node_t> const& dag = opt.get_dag();
  int total = 0;
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    // it need not be the case that there is only one domain for duration,
    // apparently.
    total += worker[nid].min() * duration[nid].min();
  }
  float utilization = 100.0 * (total / (1.0*opt.get_num_workers()*makespan.val()));

  os << "Partition makespan, utilization:  " << makespan.val() << ",  ";
  os << utilization << "%" << std::endl;

  //for(nid_t nid = 0; nid != dag.size(); ++nid) {
  //  node_t const& node = dag[nid];

  //  std::cout << nid << ": " << node << " ";

  //  std::cout << "% t = " << start[nid];
  //  std::cout << "% d = " << duration[nid];

  //  std::cout << " % b = ";
  //  nid_t compute_nid = opt.get_compute_nid(nid);
  //  vector<int> inc_part = opt.get_which_partition(
  //    compute_nid,
  //    partitions[compute_nid].val());

  //  vector<int> part;
  //  if(node.type == node_t::node_type::join) {
  //    part = inc_part;
  //  } else {
  //    part = opt.get_out_from_compute(inc_part, nid);
  //  }
  //  for(int p: part) {
  //    std::cout << p << " ";
  //  }

  //  std::cout << std::endl;
  //}
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

// This is a bit annoying. Each node in the dag is either
// input/reblock/join/agg, but the dag is structured into "super nodes"
// where a super node is either an input or a [reblocks]->join->agg.
// If any node of a super node is included, all the nodes of that super node
// need to be inlcuded.
vector<nid_t> Partition::_get_nids(int n) const {
  vector<node_t> const& dag = opt.get_dag();

  // include all the inputs right off the bat
  vector<nid_t> const& nids =
    opt.breadth_order()     ?
    opt.super_breadth_dag_order() :
    opt.super_depth_dag_order()   ;

  //std::cout << "-----------------------" << std::endl;
  //for(nid_t nid: nids) {
  //  std::cout << nid << "            " << dag[nid] << std::endl;
  //}
  //std::cout << "-----------------------" << std::endl;

  vector<nid_t> ret;
  int cnt = 0;
  for(int idx = 0; idx < nids.size() && idx < n; ++idx) {
    nid_t const& compute_nid = nids[idx];

    for(nid_t nid: opt.super(compute_nid)) {
      ret.push_back(nid);
    }
  }

  return ret;
}

void Partition::write_usage(std::ostream& out) const {
  vector<node_t> const& dag = opt.get_dag();

  // 144 * 50000 = not that big when you have >100GB of ram...
  // for each worker, for each time point, is this thread working
  vector<vector<char>> vs(opt.get_num_workers(), vector<char>(makespan.val(), 0));
  auto get_worker = [&vs](int beg, int end) {
    for(int w = 0; w != vs.size(); ++w) {
      bool valid = true;
      for(int j = beg; j != end && valid; ++j) {
        if(vs[w][j]) {
          valid = false;
        }
      }
      if(valid) {
        std::fill(vs[w].begin() + beg, vs[w].begin() + end, 1);
        return w;
      }
    }
    //throw std::runtime_error("this shouldn't happen"); (but it does)
    return -1;
  };
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    // NOTE: not all domains have to be fixed at this point(?)
    //       using the minimum.
    int s = start[nid].min();
    int e = end[nid].min();
    int n = worker[nid].min();
    for(int i = 0; i != n; ++i) {
      out << s << ","
          << e << ","
          << get_worker(s,e) << ","
          << dag[nid].type << std::endl;
    }
  }
}

// get the output partition, only valid if the corresponding
// partitions variable has a domain of 1, that is if a solution
// has been found
vector<int> Partition::get_set_partition(nid_t nid) const {
  nid_t compute_nid = opt.get_compute_nid(nid);
  vector<int> compute_partition = opt.get_which_partition(
    compute_nid,
    // this is only valid if the domain of partitions[nid] is size 1
    partitions[compute_nid].val());

  return opt.get_node_incident(compute_partition, nid);
}

}}

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include "run.h"

namespace bbts { namespace dag {

using namespace Gecode;

// NOTE: Search::Options is not a proper data type, it contains pointers to stuff
//       that get modified and deleted. So call this function to get a "fresh"
//       object and don't use search options more than once.
Search::Options build_search_options(partition_options_t const& opt) {
  Search::Options so;

  // Basically copying runMeta in driver/Script.hpp for the
  // solution case.
  //
  // The interaction between the options is not completely clear.

  so.threads = opt.search_compute_threads();

  so.c_d = Search::Config::c_d;
  so.a_d = Search::Config::a_d;
  so.d_l = Search::Config::d_l;

  // stop on only interrupt or after the time limit
  //so.stop = Driver::CombinedStop::create(
  //            0u , 0u, opt.search_time_per_cover(), true);
  so.stop = Driver::CombinedStop::create(
                0u , 0u, opt.search_time_per_cover(), false);

  so.cutoff  = Search::Cutoff::luby(opt.search_restart_scale());

  so.clone   = false;

  // When this isn't needed, gecode calls installCtrlHandler(false)...
  //Driver::CombinedStop::installCtrlHandler(true);

  return so;
}


Partition* _run(Partition* init, partition_options_t const& opt) {
  RBS<Partition, BAB> engine(init, build_search_options(opt));

  Partition* ret = nullptr;

  do {
    Partition* ex = engine.next();
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

vector<run_info_t> run(partition_options_t const& opt)
{
  auto init_info = Partition::build_init(opt);

  Partition* partition = _run(new Partition(opt, init_info), opt);

  while(partition && !partition->covers_all()) {
    // get the next cover
    Partition* next_init = new Partition(*partition, init_info);

    // reset partition
    delete partition;
    partition = nullptr;

    // solve for this cover
    partition = _run(next_init, opt);
  }

  if(!partition) {
    throw std::runtime_error("Could not find a solution!");
  }

  // For each node, get the start time and the partitioning
  vector<run_info_t> ret;
  ret.reserve(opt.get_dag().size());
  for(nid_t nid = 0; nid != opt.get_dag().size(); ++nid) {
    ret.push_back(run_info_t{
      .priority = partition->get_set_start_time(nid),
      .blocking = partition->get_set_partition(nid)
    });
  }

  delete partition;

  return ret;
}

}}

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include "partition.h"

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

vector<partition_info_t> run_partition(partition_options_t const& opt)
{
  vector<partition_info_t> ret(opt.size());

  Partition* partition = _run(new Partition(opt, ret), opt);

  if(!partition) {
    throw std::runtime_error("Could not find a solution!");
  } else {
    for(nid_t const& nid: partition->covering()) {
      ret[nid] = partition_info_t(
        partition->get_partition(nid),
        partition->get_start(nid),
        partition->get_duration(nid),
        partition->get_worker(nid),
        partition->get_unit(nid));
    }
  }

  while(partition->num_covered() != opt.size()) {
    std::cout << "covered: " << partition->num_covered()
      << " / " << opt.size() << "\n"; //".           ";

    Partition* tmp = new Partition(opt, ret);
    std::cout << "Covering to " << tmp->num_covered() << "." << std::endl;

    delete partition;

    partition = _run(tmp, opt);

    if(!partition) {
      throw std::runtime_error("Could not find a solution!");
    } else {
      for(nid_t const& nid: partition->covering()) {
        ret[nid] = partition_info_t(
          partition->get_partition(nid),
          partition->get_start(nid),
          partition->get_duration(nid),
          partition->get_worker(nid),
          partition->get_unit(nid));
      }
    }
  }

  delete partition;

  return ret;
}

}}

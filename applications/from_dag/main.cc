#include <iostream>

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include "../../src/server/node.h"

#include "dag.h"
#include "parse.h"
#include "partition/partition.h"
#include "generate.h"

using namespace bbts::dag;
using namespace Gecode;

class PartitionOptions : public Gecode::BaseOptions {
  Driver::IplOption         _ipl;
  Driver::UnsignedIntOption _restart_scale;
  Driver::UnsignedIntOption _seed;
  Driver::StringValueOption _dag_file;
  Driver::UnsignedIntOption _num_workers;
  Driver::DoubleOption _flops_per_time;
  Driver::IntOption _min_cost;
  Driver::BoolOption _breadth_order;
  Driver::BoolOption _cover;
  Driver::UnsignedIntOption _cover_size;
  Driver::UnsignedIntOption _search_compute_threads;
  Driver::UnsignedIntOption _search_restart_scale;
  Driver::UnsignedIntOption _search_time_per_cover;
  Driver::StringValueOption _output_file;
  Driver::StringValueOption _usage_file;

  vector<node_t> dag;
public:

  // Initialize options
  PartitionOptions(const char* n):
    BaseOptions(n),
    _restart_scale("restart-scale","scale factor for restart sequence",150),
    _seed("seed","random number generator seed",1U),
    _dag_file("dag-file", "File containing the dag to partition", "dag/matmul.dag"),
    _num_workers("num-workers", "Number of workers", 24),
    _flops_per_time("flops-per-time", "Number of flops per unit of time", 1e8),
    _min_cost("min-cost", "defaults to number of workers", -1),
    _breadth_order("breadth-order", "Additional constraint on order of nodes", false),
    _cover("cover", "Whether or not to use the cover algorithm", false),
    _cover_size("cover-size", "Number of nodes to consider at a time in iterative solve", 20),
    _search_compute_threads("search-compute-threads", "Number of threads for gecode to search with", 24),
    _search_restart_scale("search-restart-scale", "Restart scale param", Search::Config::slice),
    _search_time_per_cover("search-time-per-cover", "How long each iteration can take, ms", 4000),
    _output_file("output-file", "write the blockings here", ""),
    _usage_file("usage-file", "write out [(start,end,thread,label)] to this file", "")
    // ^ Here, a 10k matrix multiply (with 1e4^3=1e12 flops) will take
    //   get_min_cost() + 10,000 units of time
  {
    add(_ipl);
    add(_seed);
    add(_restart_scale);
    add(_dag_file);
    add(_num_workers);
    add(_min_cost);
    add(_flops_per_time);
    add(_breadth_order);
    add(_cover);
    add(_cover_size);
    add(_search_compute_threads);
    add(_search_restart_scale);
    add(_search_time_per_cover);
    add(_output_file);
    add(_usage_file);
  }

  // Parse options from arguments
  partition_options_t parse(int& argc, char* argv[])
  {
    BaseOptions::parse(argc,argv);

    return partition_options_t(
      parse_dag(_dag_file.value()),
      _ipl.value(),
      static_cast<int>(_restart_scale.value()),
      static_cast<int>(_seed.value()),
      static_cast<int>(_num_workers.value()),
      _flops_per_time.value(),
      _min_cost.value(),
      _breadth_order.value(),
      _cover.value(),
      static_cast<int>(_cover_size.value()),
      static_cast<int>(_search_compute_threads.value()),
      static_cast<int>(_search_restart_scale.value()),
      static_cast<int>(_search_time_per_cover.value()));
  }
};

partition_options_t get_options(int argc, char** argv) {
  PartitionOptions opt("from_dag");
  return opt.parse(argc, argv);
}

vector<bbts::ud_impl_id_t> load_cutensor_lib(
  bbts::node_t &node,
  const std::string &file_path)
{
  // try to open the file
  std::ifstream in(file_path, std::ifstream::ate | std::ifstream::binary);

  if(in.fail()) {
    throw std::runtime_error("could not load library");
  }

  auto file_len = (size_t) in.tellg();
  in.seekg (0, std::ifstream::beg);

  auto file_bytes = new char[file_len];
  in.readsome(file_bytes, file_len);

  auto [did_load, message] = node.load_shared_library(file_bytes, file_len);
  delete[] file_bytes;

  if(!did_load) {
    throw std::runtime_error(message);
  }

  vector<bbts::ud_impl_id_t> ret;
  ret.reserve(8);
  auto insert = [&](std::string name, int inn, int out) {
    auto matcher = node._udf_manager->get_matcher_for(name);
    auto ud = matcher->findMatch(
      vector<std::string>(inn, "cutensor"),
      vector<std::string>(out, "cutensor"),
      false);
    if(!ud) {
      throw std::runtime_error("could not get the ud");
    }
    ret.push_back(ud->impl_id);
  };

  // 0  init
  // 1  expand
  // 2  contraction
  // 3  reduction
  // 4  ew
  // 5  ewb
  // 6  dropout
  // 7  ewb_castable
  insert("init",         0, 1);
  insert("expand",       1, 1);
  insert("contraction",  2, 1);
  insert("reduction",    1, 1);
  insert("ew",           1, 1);
  insert("ewb",          2, 1);
  insert("dropout",      1, 1);
  insert("ewb_castable", 2, 1);

  return ret;
}

void run_commands(
  bbts::node_t& node,
  std::vector<bbts::command_ptr_t>& cmds,
  std::string loaded,
  std::string ran)
{
  auto [success_load, msg_load] = node.load_commands(cmds);

  if(!success_load) {
    throw std::runtime_error(msg_load);
  }
  std::cout << loaded << std::endl;
  std::cout << msg_load << std::endl;

  // run all the commands
  auto [success_run, msg_run] = node.run_commands();

  if(!success_run) {
    throw std::runtime_error(msg_run);
  }
  std::cout << ran << std::endl;
  std::cout << msg_run << std::endl;
};


int main(int argc, char **argv)
{
  auto options = get_options(argc, argv);

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(
          bbts::node_config_t{.argc=argc, .argv = argv, .num_threads = 24});

  // create the node
  bbts::node_t node(config);

  // init the node
  node.init();

  // sync everything
  node.sync();

  // kick off the prompt
  std::thread t;
  if (node.get_rank() == 0) {
    t = std::thread([&]()
    {
      auto uds = load_cutensor_lib(node, "PROJECT_SOURCE_DIR/libraries/libbarbcu.so");

      auto partition_info = run_partition(options);

      int num_nodes = 1; // TODO fix this
      auto [input_cmds, run_cmds] = generate_commands(
        options.get_dag(),
        partition_info,
        [&uds](int which){ return uds[which]; },
        num_nodes);

      run_commands(node, input_cmds,   "Loaded input commands",   "Ran input commands");
      run_commands(node, run_cmds, "Loaded compute commands", "Ran compute commands");
    });
  }

  // the node
  node.run();

  // wait for the prompt to finish
  if (node.get_rank() == 0) { t.join();}

  return 0;
}

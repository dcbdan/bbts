#include <iostream>

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include "../../src/server/node.h"

#include "dag.h"
#include "parse.h"
#include "partition/partition.h"
#include "generate.h"

#include "print_table.h"

#include "../../src/commands/command.h"

//https://stackoverflow.com/questions/3682773/pass-an-absolute-path-as-preprocessor-directive-on-compiler-command-line
#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

using namespace bbts::dag;
using namespace Gecode;

class PartitionOptions : public Gecode::BaseOptions {
  Driver::IplOption         _ipl;
  Driver::UnsignedIntOption _restart_scale;
  Driver::UnsignedIntOption _seed;
  Driver::StringValueOption _dag_file;
  Driver::UnsignedIntOption _num_workers;
  Driver::DoubleOption _flops_per_time;
  Driver::DoubleOption _bytes_per_time;
  Driver::IntOption _min_cost;
  Driver::IntOption _input_bytes_multiplier;
  Driver::IntOption _output_bytes_multiplier;
  Driver::IntOption _flops_multiplier;
  Driver::IntOption _max_units;
  Driver::IntOption _min_units;
  Driver::BoolOption _disallow_barrier_reblock;
  Driver::UnsignedIntOption _search_compute_threads;
  Driver::UnsignedIntOption _search_restart_scale;
  Driver::UnsignedIntOption _search_time_per_cover;
  Driver::UnsignedIntOption _cover_size;

  bool _called_help;
public:
  void help() { _called_help = true; Gecode::BaseOptions::help(); }
  bool called_help() const { return _called_help; }

  // Initialize options
  PartitionOptions(const char* n):
    BaseOptions(n),
    _restart_scale("restart-scale","scale factor for restart sequence",150),
    _seed("seed","random number generator seed",1U),
    _dag_file("dag-file", "File containing the dag to partition", "matmul.dag"),
    _num_workers("num-workers", "Number of workers", 24),
    _flops_per_time("flops-per-time", "flops thinger", 1e8),
    _bytes_per_time("bytes-per-time", "bytes thinger", 1e8),
    _min_cost("m-intercept", "...", 0),
    _input_bytes_multiplier("m-input-bytes", "...", 1),
    _output_bytes_multiplier("m-output-bytes", "...", 1),
    _flops_multiplier("m-flops", "...", 1),
    _max_units("max-units", "...", 100000),
    _min_units("min-units", "...", -1),
    _disallow_barrier_reblock("disallow-barrier-reblock", "...", false),
    _search_compute_threads("search-compute-threads", "Number of threads for gecode to search with", 24),
    _search_restart_scale("search-restart-scale", "Restart scale param", Search::Config::slice),
    _search_time_per_cover("search-time-per-cover", "How long each iteration can take, ms", 4000),
    _cover_size("cover-size", "...", 20),
    _called_help(false)
  {
    add(_ipl);
    add(_seed);
    add(_restart_scale);
    add(_dag_file);
    add(_num_workers);

    add(_min_cost);
    add(_flops_per_time);
    add(_bytes_per_time);
    add(_input_bytes_multiplier);
    add(_output_bytes_multiplier);

    add(_flops_multiplier);
    add(_max_units);
    add(_min_units);
    add(_disallow_barrier_reblock);

    add(_search_compute_threads);
    add(_search_restart_scale);
    add(_search_time_per_cover);

    add(_cover_size);
  }

  // Parse options from arguments
  void parse(int& argc, char* argv[])
  {
    BaseOptions::parse(argc,argv);
  }

  std::shared_ptr<partition_options_t>
  get_options() {
    return std::make_shared<partition_options_t>(
      parse_dag(_dag_file.value()),
      _ipl.value(),
      static_cast<int>(_restart_scale.value()),
      static_cast<int>(_seed.value()),
      static_cast<int>(_num_workers.value()),
      _flops_per_time.value(),
      _bytes_per_time.value(),
      _min_cost.value(),
      _input_bytes_multiplier.value(),
      _output_bytes_multiplier.value(),
      _flops_multiplier.value(),
      _max_units.value(),
      _min_units.value(),
      _disallow_barrier_reblock.value(),
      static_cast<int>(_search_compute_threads.value()),
      static_cast<int>(_search_restart_scale.value()),
      static_cast<int>(_search_time_per_cover.value()),
      static_cast<int>(_cover_size.value()));
  }
};

std::shared_ptr<partition_options_t> get_options(int argc, char** argv) {
  PartitionOptions opt("from_dag");
  opt.parse(argc, argv);
  // here, the parse is unsuccessful if and only if help was called,
  // which is not great.
  if(opt.called_help()) {
    return nullptr;
  } else {
    return opt.get_options();
  }
}

ud_info_t load_kernel_lib(
  ::bbts::node_t &node,
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

  auto get = [&](std::string name, int inn, int out) {
    auto matcher = node._udf_manager->get_matcher_for(name);
    auto ud = matcher->findMatch(
      vector<std::string>(inn, "cutensor"),
      vector<std::string>(out, "cutensor"),
      false);
    if(!ud) {
      throw std::runtime_error("could not get the ud");
    }
    return ud->impl_id;
  };

  ud_info_t ret{
    .init                 = get("init",         0, 1),
    .expand               = get("expand",       1, 1),
    .castable_elementwise = get("castable_ew",  2, 1),
    .permute              = get("permute",      1, 1),
    .batch_matmul         = get("batch_matmul", 2, 1),
    .reduction            = get("reduction",    1, 1),
    .unary_elementwise    = get("unary_ew",     1, 1),
    .binary_elementwise   = get("binary_ew",    2, 1)
  };

  return ret;
}

void verbose(std::ostream &out, ::bbts::node_t &node, bool val) {
  // run all the commands
  auto [did_load, message] = node.set_verbose(val);

  // did we fail
  if(!did_load) {
    throw std::runtime_error("could not set verbose");
  }
}

void run_commands(
  ::bbts::node_t& node,
  std::vector<::bbts::command_ptr_t>& cmds,
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

tuple<bool, vector<int>> parse_repl_line(std::string const& str)
{
  std::stringstream s(str);
  int i;
  vector<int> ret;

  while(!s.eof()) {
    if(s.peek() == std::stringstream::traits_type::eof()) {
      if(ret.size() > 0) {
        return {true, ret};
      } else {
        return {false, {}};
      }
    }

    char c = s.peek();

    if(std::isspace(c)) {
      s.get();
      continue;
    }

    if(std::isdigit(static_cast<unsigned char>(c))) {
      s >> i;
      ret.push_back(i);
    } else {
      return {false, {}};
    }
  }
  return {true, ret};
}

// A very bad finicky repl..... There are two things it does:
// - given an (nid, bid) pair, print out the corresponding tensor.
// - or given an empty line, print out the dag + partitioning.
void repl(::bbts::node_t& node, generate_commands_t& g) {
  auto print_dag = [&]() {
    vector<nid_t> idxs = g.priority_dag_order();
    for(nid_t nid: idxs) {
      std::cout << nid;
      if(!g[nid].is_no_op) {
        std::cout << "    *";
      } else {
        std::cout << "     ";
      }
      std::cout << ": ";

      g[nid].print(std::cout);
      std::cout << std::endl;
    }
  };

  print_dag();

  std::string s;
  while(true) {
    std::cout << "##>> ";
    std::getline(std::cin, s);

    if(s == "") {
      print_dag();
      continue;
    }

    auto [success, ints] = parse_repl_line(s);
    if(success) {
      nid_t nid = ints[0];
      if(nid < 0 || nid >= g.size()) {
        std::cout << "invalid node id" << std::endl;
        continue;
      }

      vector<int> bid(ints.begin() + 1, ints.end());
      auto& relation = g[nid];
      if(bid.size() != relation.partition.size()) {
        std::cout << "invalid block id" << std::endl;
        continue;
      }

      int which_bad = -1;
      for(int r = 0; r != relation.partition.size(); ++r) {
        if(bid[r] < 0 || bid[r] >= relation.partition[r]) {
          which_bad = r;
          break;
        }
      }
      if(which_bad >= 0) {
        std::cout << "invalid block id, index " << which_bad << std::endl;
        continue;
      }

      auto [tid, _0] = g[nid][bid];
      auto [got_it, msg] = node.print_tensor_info(tid);
      if(!got_it) {
        std::cout << "Could not get tensor id " << tid << ". " << msg << std::endl;
        continue;
      }
      std::cout << msg << std::endl;
    } else {
      std::cout << "exiting..." << std::endl;
      return;
    }
  }
}

void cmd_to_table(table_t& table, bbts::command_t const& cmd)
{
  table << cmd.id;

  if(cmd.is_apply()) {
    table << "APPLY";
  } else if(cmd.is_move()) {
    table << "MOVE";
  } else if(cmd.is_delete()) {
    table << "DELETE";
  } else if(cmd.is_broadcast()) {
    table << "BROADCAST";
  } else if(cmd.is_reduce()) {
    table << "REDUCE";
  } else if(cmd.is_touch()) {
    table << "TOUCH";
  } else {
    table << "?????";
  }

  table << (cmd.fun_id.ud_id);

  std::vector<bbts::tid_t> inn_tids;
  std::vector<bbts::node_id_t> inn_locs;
  for(int i = 0; i != cmd.get_num_inputs(); ++i) {
    auto [tid,loc] = cmd.get_input(i);
    inn_tids.push_back(tid);
    inn_locs.push_back(loc);
  }
  std::vector<bbts::tid_t> out_tids;
  std::vector<bbts::node_id_t> out_locs;
  for(int i = 0; i != cmd.get_num_outputs(); ++i) {
    auto [tid,loc] = cmd.get_output(i);
    out_tids.push_back(tid);
    out_locs.push_back(loc);
  }

  table << inn_tids << out_tids << inn_locs << out_locs << table.endl;
}

int main(int argc, char **argv)
{
  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t(argc, argv));

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
      //verbose(std::cout, node, true);

      auto options_ptr = get_options(config->argc, config->argv);
      if(!options_ptr) {
        return;
      }
      partition_options_t const& options = *options_ptr;

      ud_info_t ud_info = load_kernel_lib(node, STRINGIZE(FROM_DAG_KERNEL_LIB));

      auto partition_info = run_partition(options);
      auto get_inc_part = [&](nid_t nid) {
        return partition_info[nid].blocking;
      };

      {
        table_t table(2);
        table << "nid" << "node" << "inputs" << "partition" << "start"
              << "unit" << "# workers" << "d" << "d:init" << "d:in" << "d:out" << "d:compute"
              << table.endl;
        for(nid_t nid = 0; nid != options.get_dag().size(); ++nid) {
          auto const& pp = partition_info[nid];
          auto cc = options.to_gecode_cost_terms(
            options.get_kernel_cost(get_inc_part, nid));
          int d = cc.init + cc.input + cc.output + cc.compute;
          table << nid;
          auto const& node = options.get_dag()[nid];
          if(node.type == node_t::node_type::input) {
            table << "I";
          } else
          if(node.type == node_t::node_type::join) {
            table << "J";
          } else
          if(node.type == node_t::node_type::reblock) {
            table << "R";
          } else {
            table << "A";
          }
          table << node.downs;
          table << pp.blocking <<
            (d == 0 ? (-1) : pp.start) << pp.unit << pp.worker << d << cc.init << cc.input << cc.output <<
            cc.compute << table.endl;
        }

        std::cout << table << std::endl;
      }

      generate_commands_t g(
        options.get_dag(),
        partition_info,
        ud_info,
        node.get_num_nodes());

      auto [input_cmds, run_cmds] = g.extract();

      //{
      //  table_t table(4);
      //  table << "" << "id" << "type" << "kernel" << "inn" << "out" << "inn locs" << "out locs" << table.endl;
      //  for(auto& cmd_ptr: input_cmds) {
      //    table << "inn";
      //    cmd_to_table(table, *cmd_ptr);
      //  }
      //  for(auto& cmd: run_cmds) {
      //    table << "out";
      //    cmd_to_table(table, *cmd);
      //  }

      //  std::cout << table << std::endl;
      //}

      run_commands(node, input_cmds, "Loaded input commands",   "Ran input commands");
      run_commands(node, run_cmds,   "Loaded compute commands", "Ran compute commands");

      //repl(node, g);

      auto [did_shutdown, message] = node.shutdown_cluster();
      if(!did_shutdown) {
        throw std::runtime_error("did not shutdown: " + message);
      }
    });
  }

  // the node
  node.run();

  // wait for the prompt to finish
  if (node.get_rank() == 0) { t.join();}

  return 0;
}


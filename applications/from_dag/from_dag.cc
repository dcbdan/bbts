#include <iostream>
#include <ostream>
#include <iterator>

#include "../../src/server/node.h"

#include "dag.h"
#include "parse.h"
#include "dpar/partition.h"
#include "generate.h"
#include "greedy_placement.h"
#include "print_table.h"

#include "../../src/commands/command.h"

//https://stackoverflow.com/questions/3682773/pass-an-absolute-path-as-preprocessor-directive-on-compiler-command-line
#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

#define DCB01(x) std::cout << __LINE__ << " " << x << std::endl

using namespace bbts::dag;

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

void print_commands(std::ostream& os, vector<bbts::command_ptr_t> const& cmds) {
  for(auto const& cmd_ptr: cmds) {
    bbts::command_t const& cmd = *cmd_ptr;
    if(cmd.type == bbts::command_t::op_type_t::APPLY) {
      os << "APPLY";
    } else
    if(cmd.type == bbts::command_t::op_type_t::REDUCE) {
      os << "REDUCE";
    } else
    if(cmd.type == bbts::command_t::op_type_t::MOVE) {
      os << "MOVE";
    } else
    if(cmd.type == bbts::command_t::op_type_t::TOUCH) {
      os << "TOUCH";
    } else {
      continue;
    }
    os << ",";
    os << cmd.get_num_inputs() << ",";
    os << cmd.get_num_outputs();
    for(int i = 0; i != cmd.get_num_inputs(); ++i) {
      auto const& [tid,node] = cmd.get_input(i);
      os << "," << tid << "," << node;
    }
    for(int i = 0; i != cmd.get_num_outputs(); ++i) {
      auto const& [tid,node] = cmd.get_output(i);
      os << "," << tid << "," << node;
    }
    os << std::endl;
  }
}

vector<vector<int>> read_vecvec_from_file(std::string filename) {
  std::ifstream f(filename);
  vector<vector<int>> ret(1);
  int i;
  while(f && !f.eof()) {
    char c = f.peek();
    if(c == '\n') {
      if(ret.back().size() != 0) {
        ret.push_back(vector<int>());
      }
      f.get();
    } else
    if(std::isdigit(static_cast<unsigned char>(c))) {
      f >> i;
      ret.back().push_back(i);
    } else {
      f.get();
    }
  }
  if(ret.size() > 0 && ret.back().size() == 0) {
    ret.pop_back();
  }
  return ret;
}

std::unordered_map<int, vector<int>> build_possible_parts(
  dag_t const& dag,
  int num_workers,
  std::string filename)
{
  std::unordered_map<int, vector<int>> ret;

  // Collect all the dimensions in this dag into dims
  std::set<int> dims;
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    for(auto const& d: dag[nid].dims) {
      dims.insert(d);
    }
  }

  // Read a vector of vector of ints from the file.
  // Assume the first item in each vec is the dim, the rest possible parts.
  // If anything fails, fine, continue on.
  for(auto xs: read_vecvec_from_file(filename)) {
    if(xs.size() == 0) {
     continue;
    }
    int d = xs[0];
    if(dims.count(d) == 0) {
      continue;
    }
    if(ret.count(d) > 0) {
      continue;
    }
    std::set<int> options;
    for(int idx = 1; idx < xs.size(); ++idx) {
      int const& p = xs[idx];
      if(d % p == 0 && p <= num_workers) {
        options.insert(p);
      }
    }
    if(options.size() > 0) {
      ret[d] = vector<int>(options.begin(), options.end());
    }
  }

  // For each dim still unassigned, collect the set of potential parts
  for(auto const& d: dims) {
    if(ret.count(d) > 0) {
      continue;
    }
    for(int x: {1,2,4,8,16,32,64,128,256}) {
      if(x <= num_workers && d % x == 0) {
        ret[d].push_back(x);
      }
    }
  }

  return ret;
}

struct partition_solution_t {
  partition_solution_t(dag_t const& dag, vector<vector<int>> const& partition):
    relations(dag, partition)
  {}

  relations_t relations;
  uint64_t cost;
  vector<vector<int>> compute_locs;
};

partition_solution_t* build_partition_info(
  dag_t const& dag,
  search_params_t const& params,
  std::unordered_map<int, vector<int>> const& possible_parts,
  int num_ranks)
{
  vector<vector<int>> partition = run_partition(
    dag,
    params,
    possible_parts);

  partition_solution_t* ret = new partition_solution_t(dag, partition);
  auto& relations = ret->relations;

  vector<vector<vector<int>>> items {
    just_computes(greedy_solve_placement(false, true,  relations, num_ranks)),
    just_computes(greedy_solve_placement(false, false, relations, num_ranks)),
    just_computes(dyn_solve_placement(relations, num_ranks)),
    dumb_solve_placement(relations, num_ranks)
  };

  vector<uint64_t> costs;
  if(num_ranks > 1) {
    for(auto const& item: items) {
      costs.push_back(total_move_cost(relations, item));
    }

    auto iter = std::min_element(costs.begin(), costs.end());
    int which = std::distance(costs.begin(), iter);

    ret->cost = *iter;
    ret->compute_locs = items[which];
  } else {
    // When there is only one rank, move costs isn't a good metric...
    // Just pick the one with the most number of no ops
    ret->cost = 1000000000;
    for(int i = 0; i != ret->relations.size(); ++i) {
      if(ret->relations[i].is_no_op()) {
        ret->cost--;
      }
    }
    ret->compute_locs = items[0]; // all 'items' are the same
  }


  // now that ret is set, just print out all the info

  {
    int num_agg = 0;
    int num_reb = 0;

    table_t table(2);
    table << "nid" << "inputs" << "partition" << "unit" << "node" << table.endl;
    for(nid_t const& nid: dag.breadth_dag_order()) {
      auto const& node = dag[nid];
      table << nid << node.downs << relations[nid].partition;

      int num_unit;
      if(relations[nid].is_no_op()) {
        num_unit = 0;
      } else {
        num_unit = product(relations[nid].partition);
      }
      table << num_unit;

      if(node.type == node_t::node_type::input) {
        table << "I";
      } else
      if(node.type == node_t::node_type::join) {
        table << "J";
      } else
      if(node.type == node_t::node_type::reblock) {
        table << "R";
        if(num_unit > 0) { num_reb++; }
      } else
      if(node.type == node_t::node_type::mergesplit) {
        table << "M";
      } else {
        table << "A";
        if(num_unit > 0) { num_agg++; }
      }
      table << table.endl;
    }
    std::cout << table << std::endl;
    std::cout << "num aggregation: " << num_agg << std::endl;
    std::cout << "num reblock:     " << num_reb << std::endl;
    std::cout << std::endl;
  }

  { // Print out the parts table
    table_t table(2);
    table << "dim " << "parts..." << table.endl;
    for(auto const& [dim, ps]: possible_parts) {
      table << dim;
      for(auto const& p: ps) {
        table << p;
      }
      table << table.endl;
    }
    std::cout << table << std::endl;
  }

  if(num_ranks > 1) { // Print out the costs
    table_t table(2);
    table << "which method" << "cost" << table.endl;
    table << "ft" << (costs[0] / 10000000) << table.endl;
    table << "ff" << (costs[1] / 10000000) << table.endl;
    table << "dy" << (costs[2] / 10000000) << table.endl;
    table << "dd" << (costs[3] / 10000000) << table.endl;
    std::cout << table << std::endl;
  }

  return ret;
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
      DCB01("A argc: " << argc);
      // USAGE:
      //   DAG FILE              [REQUIRED]
      //   NUM VIRTUAL WORKERS   [REQUIRED]
      //   MAX DEPTH
      //   F_MIN F_MAX
      //   B_MIN B_MAX
      //   REBLOCK INTERCEPT
      //   BARRIER REBLOCK INTERCEPT
      //   POSSIBLE_PARTS_FILE
      if(argc < 3) {
        throw std::runtime_error("not enough arguments");
      }

      std::string file = argv[1];
      DCB01(argv[1]);

      vector<node_t> nodes = parse_dag(file);

      DCB01("superizing next!");
      superize(nodes);
      dag_t dag(nodes);
      DCB01("did the superize!");

      search_params_t params {
        .num_workers                  = 0,
        .max_depth                    = 3,
        .all_parts                    = true,
        .include_outside_up_reblock   = false,
        .include_outside_down_reblock = true,
        .flops_scale_min              = 1,
        .flops_scale_max              = 100,
        .bytes_scale_min              = 1,
        .bytes_scale_max              = 100,
        .reblock_intercept            = 1,
        .barrier_reblock_intercept    = 2
      };
      std::string possible_parts_file = "possible_parts";

      params.num_workers =                      std::stoi(argv[2]);
      DCB01("num workers: " << params.num_workers);

      DCB01(argv[3]);
      if(argc > 3)  { params.max_depth                  = std::stoi(argv[3]); }

      DCB01("a");
      if(argc > 4)  { params.flops_scale_min            = std::stoi(argv[4]); }
      DCB01("b");
      if(argc > 5)  { params.flops_scale_max            = std::stoi(argv[5]); }

      DCB01("c");
      if(argc > 6)  { params.bytes_scale_min            = std::stoi(argv[6]); }
      DCB01("d");
      if(argc > 7)  { params.bytes_scale_max            = std::stoi(argv[7]); }

      DCB01("e");
      if(argc > 8)  { params.reblock_intercept          = std::stoi(argv[8]); }
      DCB01("f");
      if(argc > 9)  { params.barrier_reblock_intercept  = std::stoi(argv[9]); }
      DCB01("g");
      if(argc > 10) { possible_parts_file               = argv[10];           }

      {
        table_t table(4);
        table << "search param" << "value" << table.endl;
        table << "num_workers        " << params.num_workers        << table.endl;
        table << "max_depth          " << params.max_depth          << table.endl;
        table << "all_parts          " << params.all_parts          << table.endl;
        table << "flops_scale_min    " << params.flops_scale_min    << table.endl;
        table << "flops_scale_max    " << params.flops_scale_max    << table.endl;
        table << "bytes_scale_min    " << params.bytes_scale_min    << table.endl;
        table << "bytes_scale_max    " << params.bytes_scale_max    << table.endl;
        table << "reblock_intercept  " << params.reblock_intercept  << table.endl;
	table << "barrier r intercpet" << params.barrier_reblock_intercept
		                                                    << table.endl;
        table << "parts file         " << possible_parts_file       << table.endl;
        std::cout << table << std::endl;
      }

      int num_ranks = node.get_num_nodes();

      vector<std::unordered_map<int, vector<int>>> parts(5);
      auto all_parts = build_possible_parts(dag, params.num_workers, possible_parts_file);
      for(auto [dim, ps]: all_parts) {
        if(ps.size() <= 4) {
          parts[0][dim] = ps;
          parts[1][dim] = ps;
          parts[2][dim] = ps;
	  parts[3][dim] = ps;
	  parts[4][dim] = ps;
        } else {
          // *  *  *  *  *
	  // 0  1  2  3
	  //    3  2  1  0
       	  auto v0 = ps[0];
          auto v1 = ps[1];
          auto v2 = ps[3];
          auto v3 = ps[4];
	  auto v4 = ps[5];
          auto m3 = ps[ps.size()-4];
          auto m2 = ps[ps.size()-3];
          auto m1 = ps[ps.size()-2];
          auto m0 = ps[ps.size()-1];
          parts[0][dim] = {v0, v1, v2, v3, m0};
          parts[1][dim] = {v0, m3, m2, m1, m0};
          parts[2][dim] = {v0, v1, m2, m1, m0};
          parts[3][dim] = {v0, m3, m2};
	  parts[4][dim] = {v0, v1, m2, m1};
        }
      }

      partition_solution_t* info = nullptr;
      for(auto const& partI: parts) {
        auto maybe = build_partition_info(dag, params, partI, num_ranks);
	      if(info == nullptr || maybe->cost < info->cost) {
	        if(info != nullptr) {
                  delete info;
	        }
                info = maybe;
	      } else {
                delete maybe;
	      }
      }

      std::cout << "Best cost: " << (info->cost / 10000000) << std::endl;

      DCB01("E load kernel lib next...");

      ud_info_t ud_info = load_kernel_lib(node, STRINGIZE(FROM_DAG_KERNEL_LIB));

      DCB01("F generate commands next...");

      DCB01("?? " << info->relations.size());
      DCB01("?# " << info->cost);

      generate_commands_t g(
        dag,
        info->relations,
        info->compute_locs,
        ud_info,
      	num_ranks);

      DCB01("!!! " << info->cost);

      auto [input_cmds, run_cmds] = g.extract();

      DCB01("G run commands next...");

      //{
      //  std::cout << "printing command dag to commands_for_js.txt" << std::endl;

      //  std::ofstream f("commands_for_js.txt");
      //  print_commands(f, input_cmds);
      //  print_commands(f, run_cmds);
      //}

      run_commands(node, input_cmds, "Loaded input commands",   "Ran input commands");
      run_commands(node, run_cmds,   "Loaded compute commands", "Ran compute commands");

      auto [did_shutdown, message] = node.shutdown_cluster();
      if(!did_shutdown) {
        throw std::runtime_error("did not shutdown: " + message);
      }

      delete info;
    });
  }

  // the node
  node.run();

  // wait for the prompt to finish
  if (node.get_rank() == 0) { t.join();}

  return 0;
}

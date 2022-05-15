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

#define DCB01(x) // std::cout << __LINE__ << " " << x << std::endl

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

std::unordered_map<int, vector<int>> build_possible_parts(
  dag_t const& dag,
  int num_workers)
{
  std::unordered_map<int, vector<int>> ret;

  // Collect all the dimensions in this dag into dims
  std::set<int> dims;
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    for(auto const& d: dag[nid].dims) {
      dims.insert(d);
    }
  }

  // For each dim, collect the set of potential parts
  for(auto const& d: dims) {
  //  if(d == 1024) {
  //    ret[d] = vector<int>{1,2,4,8,16,32,64,128,256};
  //  } else {
  //    ret[d] = vector<int>{1};
  //  }
    for(int x: {1,2,256,128,64,32,16,8,4}) {
      if(x <= num_workers && d % x == 0 && ret[d].size() < 3) {
        ret[d].push_back(x);
      }
    }
    std::cout << ret[d] << std::endl;
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
      //   REBLOCK MULTIPLIER
      //   BARRIER REBLOCK MULTIPLIER
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
        .reblock_multiplier           = 1,
        .barrier_reblock_multiplier   = 2
      };

      params.num_workers =                      std::stoi(argv[2]);
      DCB01("num workers: " << params.num_workers);

      if(argc > 3) { params.max_depth                  = std::stoi(argv[3]); }

      if(argc > 4) { params.flops_scale_min            = std::stoi(argv[4]); }
      if(argc > 5) { params.flops_scale_max            = std::stoi(argv[5]); }

      if(argc > 6) { params.bytes_scale_min            = std::stoi(argv[6]); }
      if(argc > 7) { params.bytes_scale_max            = std::stoi(argv[7]); }

      if(argc > 8) { params.reblock_multiplier         = std::stoi(argv[8]); }
      if(argc > 9) { params.barrier_reblock_multiplier = std::stoi(argv[9]); }

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
        table << "reblock_multiplier " << params.reblock_multiplier << table.endl;
        std::cout << table << std::endl;
      }

      DCB01("B run_partition next...");

      vector<vector<int>> partition = run_partition(
        dag,
        params,
        build_possible_parts(dag, params.num_workers));

      DCB01("C build relations next...");

      relations_t relations(dag, partition);

      DCB01("D table next...");

      {
        int num_agg = 0;
        int num_reb = 0;

        table_t table(2);
        table << "nid" << "inputs" << "partition" << "unit" << "node" << table.endl;
        for(nid_t const& nid: dag.breadth_dag_order()) {
          auto const& node = dag[nid];
          table << nid << node.downs << partition[nid];

          int num_unit;
          if(relations[nid].is_no_op()) {
            num_unit = 0;
          } else {
            num_unit = product(partition[nid]);
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

      DCB01("D compute_locs next...");

      vector<vector<int>> compute_locs;

      {
        vector<vector<vector<int>>> items;
        items.resize(3);

        {
          // The first bool: in each relation, should the minimum move costed block be chosen, or just
          //                 do in order
          // The second bool: should the cost of outputs be included
          //
          // If first bool is true, scales n^2 where n is the most number of blocks in all relations.
          auto x0 = greedy_solve_placement(false, true,  relations, node.get_num_nodes());
          auto x1 = greedy_solve_placement(false, false, relations, node.get_num_nodes());

          items[0] = just_computes(x0);
          items[1] = just_computes(x1);
        }

        // A round robin placement to each relation
        items[2] = dumb_solve_placement(relations, node.get_num_nodes());

        uint64_t cost_ft = total_move_cost(relations, items[0]);
        uint64_t cost_ff = total_move_cost(relations, items[1]);
        uint64_t cost_dd = total_move_cost(relations, items[2]);

        table_t table(2);
        table << "which method" << "cost" << table.endl;
        table << "ft" << cost_ft << table.endl;
        table << "ff" << cost_ff << table.endl;
        table << "dd" << cost_dd << table.endl;
        std::cout << table;

        vector<uint64_t> costs =
          {
            cost_ft,
            cost_ff,
            cost_dd
          };

        auto iter = std::min_element(costs.begin(), costs.end());
        int which = std::distance(costs.begin(), iter);
        std::cout << "which: " << which << std::endl;
        compute_locs = items[which];
      }

      DCB01("E load kernel lib next...");

      ud_info_t ud_info = load_kernel_lib(node, STRINGIZE(FROM_DAG_KERNEL_LIB));

      DCB01("F generate commands next...");

      generate_commands_t g(
        dag,
        relations,
        compute_locs,
        ud_info,
        node.get_num_nodes());

      auto [input_cmds, run_cmds] = g.extract();

      DCB01("G run commands next...");

      {
        std::cout << "printing command dag to commands_for_js.txt" << std::endl;

        std::ofstream f("commands_for_js.txt");
        print_commands(f, input_cmds);
        print_commands(f, run_cmds);
      }

      run_commands(node, input_cmds, "Loaded input commands",   "Ran input commands");
      run_commands(node, run_cmds,   "Loaded compute commands", "Ran compute commands");

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

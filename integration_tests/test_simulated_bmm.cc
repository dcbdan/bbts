#include <map>
#include <thread>
#include <chrono>
#include "../src/operations/move_op.h"
#include "../src/operations/reduce_op.h"
#include "../src/commands/reservation_station.h"
#include "../src/commands/tensor_notifier.h"
#include "../src/commands/command_runner.h"
#include "../src/server/node.h"
#include "../src/server/static_config.h"

using namespace bbts;
using namespace std::chrono;

using index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<node_id_t, tid_t>>;
using to_agg_index_t = std::map<std::tuple<int32_t, int32_t>, std::vector<std::tuple<tid_t, node_id_t>>>;

// creates the matrix tensors on this node
index_t create_matrix_tensors(char matrix,
                              bbts::node_t &node,
                              int n,
                              int split,
                              int &cur_tid,
                              std::vector<command_ptr_t> &_cmds) {

  // the index
  index_t index;

  // block size
  uint32_t block_size = n / split;

  // get the udf manager
  auto udm = node._udf_manager;

  // get the rank
  auto my_rank = node._comm->get_rank();

  // create all the rows an columns we need
  auto hash_fn = std::hash<int>();
  for (int row_id = 0; row_id < split; ++row_id) {
    for (int col_id = 0; col_id < split; ++col_id) {

      // check if this block is on this node
      auto target_node = static_cast<node_id_t>(hash_fn(row_id * split + col_id) % node.get_num_nodes());

      // return me that matcher for matrix addition
      auto matcher = udm->get_matcher_for("uniform");

      // get the ud object
      auto ud = matcher->findMatch({}, {"dense"}, false);

      // set the index
      index[{row_id, col_id}] = {target_node, cur_tid};

      // store the command
      _cmds.emplace_back(command_t::create_apply(_cmds.size(),
                                                 ud->impl_id,
                                                 false,
                                                 {command_param_t{.u = block_size},
                                                         command_param_t{.u = block_size},
                                                         command_param_t{.f = 0.0f},
                                                         command_param_t{.f = 1.0f}},
                                                 {},
                                                 {command_t::tid_node_id_t{.tid = cur_tid, .node = target_node}}));

      std::cout << "UNIFORM(matrix=" << matrix << ", tensor=(" << row_id << ", " << col_id << "), tid=" << cur_tid
                << " , node=" << my_rank
                << ")\n";

      // go to the next one
      cur_tid++;
    }
  }

  // return the index
  return std::move(index);
}

// create all the broadcast commands
void create_broadcast(size_t num_nodes,
                      size_t split,
                      index_t &mat_locs,
                      std::vector<command_ptr_t> &_cmds,
                      std::vector<std::vector<int32_t>> &to_del) {


  // no need to move here
  std::vector<command_t::tid_node_id_t> outputs;

  // do the shuffle
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // get the tid and the node of this block
      auto &[node, tid] = mat_locs[{rowID, colID}];

      // set all the nodes we need to broadcast to
      outputs.clear();
      for(node_id_t n = 0; n < num_nodes; ++n) {
        if(n != node) {
          outputs.push_back(command_t::tid_node_id_t{.tid = tid, .node = n});
          to_del[n].push_back(tid);
        }
      }

      // create the broadcast
      _cmds.emplace_back(command_t::create_broadcast(_cmds.size(), command_t::tid_node_id_t{.tid = tid, .node = node}, outputs));
    }
  }
}

// create the shuffle
template<class fun>
void create_shuffle(size_t num_nodes,
                    size_t split,
                    fun fn,
                    index_t &mat_locs,
                    std::vector<command_ptr_t> &_cmds,
                    std::vector<std::vector<int32_t>> &to_del) {


  // do the shuffle
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // get the tid and the node of this block
      auto &[node, tid] = mat_locs[{rowID, colID}];

      // no need to move here
      auto target_node = (node_id_t) fn(rowID, colID, num_nodes);
      if (node == target_node) {
        continue;
      }

      // move it
      _cmds.emplace_back(command_t::create_move(_cmds.size(),
                                                command_t::tid_node_id_t{.tid = tid, .node = node},
                                                command_t::tid_node_id_t{.tid = tid, .node = target_node}));


      // mark that we need to delete it later
      to_del[target_node].push_back(tid);
    }
  }
}

template<class fun>
to_agg_index_t create_multiply(fun fn,
                               const udf_manager_ptr &udm,
                               size_t split,
                               size_t num_nodes,
                               index_t a_mat,
                               index_t b_mat,
                               int32_t &tid_offset,
                               std::vector<command_ptr_t> &_cmds,
                               std::vector<std::vector<int32_t>> &to_del) {

  // create all the multiply commands
  std::map<std::tuple<int32_t, int32_t>, std::vector<std::tuple<tid_t, node_id_t>>> multiplies;

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_mult");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false);

  // generate the applies to do the multiplication
  for (int32_t i = 0; i < split; ++i) {
    for (int32_t j = 0; j < split; ++j) {
      for (int32_t k = 0; k < split; ++k) {

        // get the tid and the node
        auto[a_node, a_tid] = a_mat[{i, k}];
        auto[b_node, b_tid] = b_mat[{k, j}];

        // get the target node
        auto target_node = (node_id_t) fn(i, k, num_nodes);

        // add the command
        _cmds.emplace_back(command_t::create_apply(_cmds.size(),
                                                   ud->impl_id,
                                                   false,
                                                   {},
                                                   { command_t::tid_node_id_t{.tid = a_tid, .node = target_node},
                                                        command_t::tid_node_id_t{.tid = b_tid, .node = target_node}},
                                                   {command_t::tid_node_id_t{.tid = tid_offset, .node = target_node}}));

        // mark that we need to delete this tensor later
        to_del[target_node].push_back(tid_offset);

        // do the multiplies
        multiplies[{i, j}].push_back({tid_offset, target_node});

        // go to next tid
        tid_offset++;
      }
    }
  }

  return std::move(multiplies);
}

void generate_aggregation(const udf_manager_ptr &udm,
                          size_t split,
                          size_t num_nodes,
                          int32_t &tid_offset,
                          to_agg_index_t &multiplies,
                          std::vector<command_ptr_t> &_cmds) {

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_add");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false);

  // create the aggregate
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // all the multiplied tensors
      auto &muls = multiplies[{rowID, colID}];

      // get the target node
      auto target_node = (node_id_t) ((rowID + colID * split) % num_nodes);

      // figure out the inputs
      std::vector<bbts::command_t::tid_node_id_t> inputs;
      for (auto &mul : muls) {
        auto &[tid, node] = mul;
        inputs.push_back({.tid = tid, .node = target_node});
      }

      // create the reduce command
      _cmds.emplace_back(command_t::create_reduce(_cmds.size(),
                                                  ud->impl_id,
                                                  false,
                                                  {},
                                                  inputs,
                                                  {command_t::tid_node_id_t{.tid = tid_offset, .node = target_node}}));

      tid_offset++;
    }
  }
}

void create_delete(size_t num_nodes,
                   std::vector<std::vector<int32_t>> &to_del, std::vector<command_ptr_t> &_cmds) {

  // prepare the removes
  for (int32_t node = 0; node < num_nodes; ++node) {


    // store the number we need to delete
    std::vector<bbts::command_t::tid_node_id_t> _inputs;
    _inputs.reserve(to_del[node].size());
    for (auto t : to_del[node]) {
      _inputs.push_back(command_t::tid_node_id_t{.tid = t, .node = node});
    }

    // remove them from node
    _cmds.emplace_back(command_t::create_delete(_cmds.size(), _inputs));
  }
}

std::vector<bbts::command_ptr_t> generate_commands(size_t split, bbts::node_t &node) {

  std::vector<bbts::command_ptr_t> commands;
  int32_t tid_offset = 0;

  auto a_idx = create_matrix_tensors('A', node, 1000, split, tid_offset, commands);
  auto b_idx = create_matrix_tensors('B', node, 1000, split, tid_offset, commands);

  // all the tensors that we need to delete
  std::vector<std::vector<int32_t>> to_del(node.get_num_nodes());

  // create the shuffle
  create_shuffle(node.get_num_nodes(),
                 split,
                 [](int32_t rowID, int32_t colID, size_t num_nodes) { return rowID % num_nodes; },
                 a_idx,
                 commands,
                 to_del);

  // create the broadcast
  create_broadcast(node.get_num_nodes(), split, b_idx, commands, to_del);

  // create the multiply commands
  auto multiplies = create_multiply([](int32_t rowID, int32_t colID, size_t num_nodes) { return rowID % num_nodes; },
                                       node._udf_manager, split, node.get_num_nodes(),
                                       a_idx, b_idx, tid_offset, commands, to_del);

  // generate the aggregation
  generate_aggregation(node._udf_manager, split, node.get_num_nodes(), tid_offset, multiplies, commands);

  // create the delete
  create_delete(node.get_num_nodes(), to_del, commands);

  return std::move(commands);
}

int main(int argc, char **argv) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t(argc, argv));
  config->num_threads = 8;

  // create the node
  bbts::node_t node(config);

  // init the node
  node.init();

  // kick of all the stuff
  std::thread t = std::thread([&]() {
    node.run();
  });

  // is this the root node
  if(node.get_rank() == 0) {

    // generate all the commands
    auto cmds = generate_commands(16, node);

    // print out all the debug info
    node.set_verbose(true);

    // load the commands
    node.load_commands(cmds);

    // start measuring time
    auto start = high_resolution_clock::now();

    // run all the commands
    node.run_commands();

    // finish measuring
    auto stop = high_resolution_clock::now();

    // shutdown the cluster
    node.shutdown_cluster();

    // log time
    std::cout << "Time to compute : " << ((double) duration_cast<nanoseconds>(stop - start).count()) / duration_cast<nanoseconds>(1s).count() << "s\n";
  }

  // finish running
  t.join();

  return 0;
}

#include <iostream>
#include <map>
#include <thread>
#include <functional>
#include <tuple>
#include <vector>
#include "../src/storage/storage.h"
#include "../src/commands/reservation_station.h"
#include "../src/commands/tensor_notifier.h"
#include "../src/commands/command_runner.h"
#include "../src/server/node.h"
#include "../src/server/static_config.h"
#include "../src/utils/expand_indexer.h"
#include "../src/tensor/builtin_formats.h"
#include "../src/ud_functions/impls/dense_iota.h"
#include "../src/ud_functions/impls/dense_expand.h"

using namespace bbts;

using std::tuple;
using std::vector;
using namespace utils::expand;

// Return a list of commands to execute as well as the tids of the output vector.
//
// The operation reblocks an input tensor the num_blocks_inn to an output tensor
// num_blocks_out. The total number of elements of each relation is given by
// num_blocks_nin*num_blocks_out*num_small.
tuple<
  vector<command_ptr_t>,
  vector<tid_t>,
  vector<tid_t>>
    generate_commands(
      node_t& node,
      int num_blocks_inn,
      int num_blocks_out,
      int num_small);

// This is a wrapper to extract tid's memory from the tos.
// Given tid's memory and the number of elements in that memory,
// run the given function.
void with_vector(
  node_t& node,
  tid_t tid,
  std::function<void(float*, uint32_t)> f);

int main(int argc, char **argv) {
  int num_blocks_inn = 4;
  int num_blocks_out = 3;
  int num_small = 100;
  if(argc == 4) {
    num_blocks_inn = stoi(std::string(argv[1]));
    num_blocks_out = stoi(std::string(argv[2]));
    num_small      = stoi(std::string(argv[3]));
  }
  std::cout << "num input blocks  " << num_blocks_inn << std::endl;
  std::cout << "num output blocks " << num_blocks_out << std::endl;
  std::cout << "dimension         " << (num_blocks_inn*num_blocks_out*num_small) << std::endl;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(
    bbts::node_config_t{.argc=argc, .argv = argv, .num_threads = 8});

  // create the node
  bbts::node_t node(config);

  // init the node
  node.init();

  // register the kernels we'll need
  node._udf_manager->register_udf(dense_iota_t::get_ud_func());
  node._udf_manager->register_udf_impl(std::make_unique<dense_iota_t>());

  node._udf_manager->register_udf(dense_expand_t::get_ud_func());
  node._udf_manager->register_udf_impl(std::make_unique<dense_expand_t>());

  // kick of all the stuff
  std::thread t = std::thread([&]() {
    node.run();
  });

  // is this the root node
  if(node.get_rank() == 0) {

    // generate all the commands
    auto [cmds, input_tids, output_tids] = generate_commands(
      node, num_blocks_inn, num_blocks_out, num_small);

    // print out all the debug info
    //node.set_verbose(true);

    // load the commands
    auto [success_load, msg_load] = node.load_commands(cmds);

    if(!success_load) {
      throw std::runtime_error(msg_load);
    }
    std::cout << "Loaded commands" << std::endl;
    std::cout << msg_load << std::endl;

    // run all the commands
    auto [success_run, msg_run] = node.run_commands();

    if(!success_run) {
      throw std::runtime_error(msg_run);
    }
    std::cout << "Ran commands" << std::endl;
    std::cout << msg_run << std::endl;

    auto check_correctness = [&](vector<int> const& tids, int expected_num_per_block)
    {
      bool correct = true;
      float val = 0.0;

      for(tid_t tid: tids) {
        //std::cout << "TID: " << tid << std::endl;
        with_vector(node, tid, [&](float* data, uint32_t num_elem) {
          if(num_elem != expected_num_per_block) {
            correct = false;
          }
          for(int idx = 0; idx != num_elem; ++idx) {
            //if(val != data[idx] && idx % 1003 == 0) {
            //  std::cout << "tid " << tid << ", idx " << idx
            //    << ": val " << data[idx] << "(val expected " << val << ")" << std::endl;
            //}
            correct = correct && (val == data[idx]);
            val++;
          }
        });
      }
      return correct;
    };

    std::cout << "Verifying input correctness" << std::endl;
    assert(input_tids.size() == num_blocks_inn);
    bool input_correct = check_correctness(input_tids, num_blocks_out*num_small);
    if(!input_correct) {
      std::cout << "INPUT WAS NOT CORRECT!" << std::endl;
    } else {
      std::cout << "Correct." << std::endl;
    }

    std::cout << "Verifying output correctness" << std::endl;
    assert(output_tids.size() == num_blocks_out);
    bool output_correct = check_correctness(output_tids, num_blocks_inn*num_small);
    if(!output_correct) {
      std::cout << "OUTPUT WAS NOT CORRECT!" << std::endl;
    } else {
      std::cout << "Correct." << std::endl;
    }

    // shutdown the cluster
    node.shutdown_cluster();
  }

  // finish running
  t.join();

  return 0;
}

void with_vector(
  node_t& node,
  tid_t tid,
  std::function<void(float*, uint32_t)> f)
{
  // reach into the node
  // to reach into the storage
  // to give a sandbox with the given tid.
  // now you have the actual memory and you can do whatever you want.
  // first convert the memory into the actual user defined tensor.
  // then figure out how many elements it has.
  // (if its a vector, num_rows or num_cols should be one, but there is no need to check)
  // then call the function with the data
  node._storage->local_transaction({tid}, {},
    [&](const storage_t::reservation_result_t &res) {
      dense_tensor_t* tensor = static_cast<dense_tensor_t*>(res.get[0].get().tensor);
      auto const& mm = tensor->meta().m();
      uint32_t num_elem = mm.num_rows * mm.num_cols;
      f(tensor->data(), num_elem);
    });
}

struct impl_t {
  impl_t(node_t& node): node(node), cur_tid(0) {}

  vector<command_ptr_t> extract_commands() { return std::move(cmds); }

  vector<tid_t> init(int num_blocks, int num_per_block) {
    vector<tid_t> out;

    auto matcher = node._udf_manager->get_matcher_for("iota");
    auto ud = matcher->findMatch({}, {"dense"}, false);

    float val = 0.0;

    for(int which_block = 0; which_block != num_blocks; ++which_block)
    {
      float val = which_block * num_per_block;

      cmds.emplace_back(command_t::create_apply(
          cmds.size(),
          ud->impl_id,
          false,
          { command_param_t{ .u = (uint32_t)num_per_block },
            command_param_t{ .u = 1u },
            command_param_t{ .f = val } },
          {},
          {command_t::tid_node_id_t{ .tid = cur_tid, .node = (int)node.get_rank() }}));

      out.push_back(cur_tid);
      cur_tid++;
    }

    return out;
  }

  vector<tid_t> reblock(vector<tid_t> const& ins, int num_per_block_inn, int num_blocks_out)
  {
    vector<tid_t> out;

    assert(ins.size() != 0);
    assert(num_blocks_out > 0);

    int num_blocks_inn = ins.size();
    int dim = num_per_block_inn * num_blocks_inn;

    expand_indexer_t indexer({num_blocks_inn}, {num_blocks_out});

    auto make_expand_params = [&](int which_block) {
      return std::vector<command_param_t>{
        command_param_t { .u = (uint32_t)dim },
        command_param_t { .u = (uint32_t)num_blocks_inn },
        command_param_t { .u = (uint32_t)num_blocks_out },
        command_param_t { .u = (uint32_t)which_block },
        command_param_t { .u = 1 },
        command_param_t { .u = 1 },
        command_param_t { .u = 1 },
        command_param_t { .u = 0 }};
    };

    auto matcher = node._udf_manager->get_matcher_for("expand");
    auto ud = matcher->findMatch({"dense"}, {"dense"}, false);

    for(int which_block = 0; which_block != num_blocks_out; ++which_block)
    {
      // this block depends on inputs s,...,e
      auto [s,e] = indexer.get_inputs({which_block})[0];
      int num_touches = e - s + 1;
      assert(num_touches > 0);

      // then issue each of the required touches
      for(int i = s; i <= e; ++i) {
        tid_t inn_tid = ins[i];

        cmds.emplace_back(command_t::create_touch(
          cmds.size(),
          ud->impl_id,
          false,
          i - s,
          num_touches,
          // the params without compact and which
          make_expand_params(which_block),
          {command_t::tid_node_id_t{ .tid = inn_tid, .node = (int)node.get_rank() }},
          {command_t::tid_node_id_t{ .tid = cur_tid, .node = (int)node.get_rank() }}
        ));
      }

      out.push_back(cur_tid);
      cur_tid++;
    }
    return out;
  }

private:
  vector<command_ptr_t> cmds;
  node_t& node;
  int cur_tid;
};

tuple<
  vector<command_ptr_t>,
  vector<tid_t>,
  vector<tid_t>>
    generate_commands(
      bbts::node_t& node,
      int num_blocks_inn,
      int num_blocks_out,
      int num_small)
{
  impl_t impl(node);

  int num_elem = num_blocks_inn*num_blocks_out*num_small;

  int num_per_block_inn = num_elem / num_blocks_inn;
  vector<tid_t> inputs = impl.init(
    num_blocks_inn,
    num_per_block_inn);
  assert(inputs.size() == num_blocks_inn);

  vector<tid_t> out = impl.reblock(inputs, num_per_block_inn, num_blocks_out);
  assert(out.size() == num_blocks_out);

  return {impl.extract_commands(), inputs, out};
}


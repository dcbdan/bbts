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
#include "../src/ud_functions/impls/dense_index.h"
#include "../src/ud_functions/impls/dense_expand.h"

using namespace bbts;

using std::tuple;
using std::vector;

using namespace utils::expand;

template <typename T>
struct matrix {
  matrix(int ni, int nj): ni(ni), nj(nj), data(ni*nj) {}

  T const& operator()(int i, int j) const { return data[i*nj+j]; }
  T      & operator()(int i, int j)       { return data[i*nj+j]; }

  int ni;
  int nj;
  vector<T> data;
};

// Return a list of commands to execute as well as the tids of the output vector.
//
// The operation reblocks an input tensor the num_blocks_inn to an output tensor
// num_blocks_out. The total number of elements of each relation is given by
// num_blocks_nin*num_blocks_out*num_small.
tuple<
  vector<command_ptr_t>, matrix<tid_t>,
  vector<command_ptr_t>, matrix<tid_t>>
    generate_commands(
      node_t& node,
      int num_blocks_inn_i, int num_blocks_inn_j,
      int num_blocks_out_i, int num_blocks_out_j,
      int num_small_i,      int num_small_j);

// This is a wrapper to extract tid's memory from the tos.
// Given tid's memory and the number of elements in that memory,
// run the given function.
void with_vector(
  node_t& node,
  tid_t tid,
  std::function<void(float*, uint32_t, uint32_t)> f);

int main(int argc, char **argv) {
  int num_blocks_inn_i = 1;
  int num_blocks_inn_j = 3;

  int num_blocks_out_i = 2;
  int num_blocks_out_j = 2;

  int num_small_i = 2;
  int num_small_j = 2;

  if(argc >= 4) {
    num_blocks_inn_i = stoi(std::string(argv[1]));
    num_blocks_inn_j = stoi(std::string(argv[2]));

    num_blocks_out_i = stoi(std::string(argv[3]));
    num_blocks_out_j = stoi(std::string(argv[4]));
  }
  if(argc >= 6) {
    num_small_i      = stoi(std::string(argv[5]));
    num_small_j      = stoi(std::string(argv[6]));
  }

  std::cout << "input blocking  " << num_blocks_inn_i << " " << num_blocks_inn_j << std::endl;
  std::cout << "output blocking " << num_blocks_out_i << " " << num_blocks_out_j << std::endl;
  std::cout << "dimension       " << (num_blocks_inn_i*num_blocks_out_i*num_small_i) << " " <<
                                     (num_blocks_inn_j*num_blocks_out_j*num_small_j) << std::endl;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(
    bbts::node_config_t{.argc=argc, .argv = argv, .num_threads = 8});

  // create the node
  bbts::node_t node(config);

  // init the node
  node.init();

  // register the kernels we'll need

  // make sure each item element is small to avoid incorrect == operator
  // ... (floating points and large integers == loss of precision)
  auto by_index = [](int i, int j){ return 1.0 * ((i + j) % 1000) + (i % 13) + 7.5; };
  node._udf_manager->register_udf(dense_index_t::get_ud_func());
  node._udf_manager->register_udf_impl(
    std::make_unique<dense_index_t>(by_index));

  node._udf_manager->register_udf(dense_expand_t::get_ud_func());
  node._udf_manager->register_udf_impl(std::make_unique<dense_expand_t>());

  // kick of all the stuff
  std::thread t = std::thread([&]() {
    node.run();
  });

  // is this the root node
  if(node.get_rank() == 0) {

    // generate all the commands
    auto [input_cmds, input_tids, reblock_cmds, output_tids] = generate_commands(
      node,
      num_blocks_inn_i, num_blocks_inn_j,
      num_blocks_out_i, num_blocks_out_j,
      num_small_i, num_small_j);

    // print out all the debug info
    //node.set_verbose(true);

    auto run_cmds = [&](
      std::vector<command_ptr_t>& cmds,
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
    run_cmds(input_cmds,   "Loaded input commands",   "Ran input commands");
    run_cmds(reblock_cmds, "Loaded reblock commands", "Ran reblock commands");

    auto check_correctness = [&](matrix<tid_t> const& tids, int expected_ki, int expected_kj)
    {
      bool correct = true;
      for(int bi = 0; bi != tids.ni; ++bi) {
      for(int bj = 0; bj != tids.nj; ++bj) {
        with_vector(node, tids(bi,bj), [&](float* data, int ki, int kj) {
          if(ki != expected_ki || kj != expected_kj) {
            correct = false;
          }
          for(int i = 0; i != ki; ++i) {
          for(int j = 0; j != kj; ++j) {
            //std::cout
            //  << "tid " << tids(bi,bj) << "  |  b [" << bi << ", " << bj << "]"
            //  << "  |  k [" << i << ", " << j << "]  |  val "
            //  << data[kj*i+j] << "  (expected val "
            //  << by_index(bi*ki + i, bj*kj + j) << ")" << std::endl;

            correct = correct && (data[kj*i + j] == by_index(bi*ki + i, bj*kj + j));
          }}
        });
      }}
      return correct;
    };

    std::cout << "Verifying input correctness" << std::endl;
    assert(input_tids.ni == num_blocks_inn_i &&
           input_tids.nj == num_blocks_inn_j);
    bool input_correct = check_correctness(
      input_tids,
      num_blocks_out_i*num_small_i,
      num_blocks_out_j*num_small_j);
    if(!input_correct) {
      std::cout << "INPUT WAS NOT CORRECT!" << std::endl;
    } else {
      std::cout << "Correct." << std::endl;
    }

    std::cout << "Verifying output correctness" << std::endl;
    assert(output_tids.ni == num_blocks_out_i &&
           output_tids.nj == num_blocks_out_j);
    bool output_correct = check_correctness(
      output_tids,
      num_blocks_inn_i*num_small_i,
      num_blocks_inn_j*num_small_j);
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
  std::function<void(float*, uint32_t, uint32_t)> f)
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
      f(tensor->data(), mm.num_rows, mm.num_cols);
    });
}

struct impl_t {
  impl_t(node_t& node): node(node), cur_tid(0) {}

  vector<command_ptr_t> extract_commands() {
    auto ret = std::move(cmds);
    cmds.resize(0);
    return std::move(ret);
  }

  matrix<tid_t> init(int nbi, int nbj, int nki, int nkj) {
    auto matcher = node._udf_manager->get_matcher_for("index");
    auto ud = matcher->findMatch({}, {"dense"}, false);

    matrix<tid_t> out(nbi, nbj);
    for(int bi = 0; bi != nbi; ++bi) {
    for(int bj = 0; bj != nbj; ++bj) {
      cmds.emplace_back(command_t::create_apply(
          cmds.size(),
          ud->impl_id,
          false,
          {
            command_param_t{ .u = (uint32_t)((bi+0)*nki) },
            command_param_t{ .u = (uint32_t)((bi+1)*nki) },
            command_param_t{ .u = (uint32_t)((bj+0)*nkj) },
            command_param_t{ .u = (uint32_t)((bj+1)*nkj) }
          },
          {},
          {command_t::tid_node_id_t{ .tid = cur_tid, .node = (int)node.get_rank() }}));

      out(bi,bj) = cur_tid;
      cur_tid++;
    }}
    return out;
  }

  matrix<tid_t> reblock(
    matrix<tid_t> const& ins,
    int ni, int nj,
    int out_nbi, int out_nbj)
  {
    int inn_nbi = ins.ni;
    int inn_nbj = ins.nj;

    matrix<tid_t> out(out_nbi, out_nbj);

    assert(inn_nbi > 0 && inn_nbj > 0 &&
           out_nbi > 0 && out_nbj > 0);

    expand_indexer_t indexer({inn_nbi, inn_nbj},
                             {out_nbi, out_nbj});

    auto make_expand_params = [&](int bi, int bj) {
      return std::vector<command_param_t>{
        command_param_t { .u = (uint32_t) ni      },
        command_param_t { .u = (uint32_t) inn_nbi },
        command_param_t { .u = (uint32_t) out_nbi },
        command_param_t { .u = (uint32_t) bi      },
        command_param_t { .u = (uint32_t) nj      },
        command_param_t { .u = (uint32_t) inn_nbj },
        command_param_t { .u = (uint32_t) out_nbj },
        command_param_t { .u = (uint32_t) bj      }};
    };

    auto matcher = node._udf_manager->get_matcher_for("expand");
    auto ud = matcher->findMatch({"dense"}, {"dense"}, false);

    for(int bi = 0; bi != out_nbi; ++bi) {
    for(int bj = 0; bj != out_nbj; ++bj) {
      auto params = make_expand_params(bi,bj);

      vector<vector<int>> _inputs_here = indexer.cartesian(
        indexer.get_inputs({bi,bj}));

      int num_touches = _inputs_here.size();
      assert(num_touches > 0);

      for(int i = 0; i != num_touches; ++i) {
        int which_i = _inputs_here[i][0];
        int which_j = _inputs_here[i][1];
        tid_t inn_tid = ins(which_i, which_j);

        cmds.emplace_back(command_t::create_touch(
          cmds.size(),
          ud->impl_id,
          false,
          i,
          num_touches,
          params,
          {command_t::tid_node_id_t{ .tid = inn_tid, .node = (int)node.get_rank() }},
          {command_t::tid_node_id_t{ .tid = cur_tid, .node = (int)node.get_rank() }}
        ));
      }
      out(bi,bj) = cur_tid;
      cur_tid++;
    }}

    return out;
  }

private:
  vector<command_ptr_t> cmds;
  node_t& node;
  int cur_tid;
};

tuple<
  vector<command_ptr_t>, matrix<tid_t>,
  vector<command_ptr_t>, matrix<tid_t>>
    generate_commands(
      bbts::node_t& node,
      int num_blocks_inn_i, int num_blocks_inn_j,
      int num_blocks_out_i, int num_blocks_out_j,
      int num_small_i,      int num_small_j)
{
  impl_t impl(node);

  int dim_i = num_blocks_inn_i * num_blocks_out_i * num_small_i;
  int dim_j = num_blocks_inn_j * num_blocks_out_j * num_small_j;

  matrix<tid_t> inputs = impl.init(
    num_blocks_inn_i, num_blocks_inn_j,
    num_blocks_out_i*num_small_i,
    num_blocks_out_j*num_small_j);
  auto input_commands = impl.extract_commands();

  assert(inputs.ni == num_blocks_inn_i &&
         inputs.nj == num_blocks_inn_j);

  matrix<tid_t> outputs = impl.reblock(
    inputs,
    dim_i, dim_j,
    num_blocks_out_i, num_blocks_out_j);
  assert(outputs.ni == num_blocks_out_i &&
         outputs.nj == num_blocks_out_j);
  auto output_commands = impl.extract_commands();

  return {std::move(input_commands),  inputs,
          std::move(output_commands), outputs};
}


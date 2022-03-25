#include "command_runner.h"
#include "../operations/move_op.h"
#include "../operations/reduce_op.h"
#include "../operations/broadcast_op.h"
#include "../operations/local_reduce_op.h"
#include <cassert>
#include <thread>

bbts::command_runner_t::command_runner_t(bbts::storage_ptr_t ts,
                                         bbts::tensor_factory_ptr_t tf,
                                         bbts::udf_manager_ptr udm,
                                         bbts::reservation_station_ptr_t rs,
                                         bbts::communicator_ptr_t comm,
                                         bbts::logger_ptr_t logger)
    : _ts(std::move(ts)), _tf(std::move(tf)), _udm(std::move(udm)),
      _rs(std::move(rs)), _comm(std::move(comm)), _logger(std::move(logger)) {}


void bbts::command_runner_t::local_move_command_runner() {

  while (true) {

    // get the command
    auto cmd = _rs->get_next_move_command();
    if (cmd == nullptr) {
      break;
    }

    // move the
    _logger->message("MOVE " + std::to_string(cmd->id) + " on my_node : " + std::to_string(_comm->get_rank()) + " Executed...\n");

    // it is a point to point move
    if(cmd->is_move()) {

      // get the size of the tensor
      auto num_bytes = _ts->get_tensor_size(cmd->get_input(0).tid);

      // set it as a command extra info
      cmd->nfo.num_bytes = num_bytes;

      // forward the command to the right nodes
      if(!_comm->op_request(cmd)) {
        throw std::runtime_error("Failed to forward the command.");
      }

      // create the move operation
      move_op_t op(*_comm, cmd->id, num_bytes, cmd->get_input(0).tid, true, *_ts, cmd->get_output(0).node);

      // do the apply
      op.apply();

      // retire the command so it knows that we have processed the tensors
      _rs->retire_command(std::move(cmd));
    }
    // it is a broadcast
    else {

      _logger->message("BROADCAST\n");

      // get the size of the tensor
      auto num_bytes = _ts->get_tensor_size(cmd->get_input(0).tid);

      // set it as a command extra info
      cmd->nfo.num_bytes = num_bytes;

      // forward the command to the right nodes
      if(!_comm->op_request(cmd)) {
        throw std::runtime_error("Failed to forward reduce command.");
      }

      // get the nodes involved
      auto nodes = cmd->get_nodes();

      // create the move operation
      broadcast_op_t op(*_comm, *_ts, nodes, cmd->id, num_bytes, cmd->get_input(0).tid);

      // do the apply
      op.apply();

      // retire the command so it knows that we have processed the tensors
      _rs->retire_command(std::move(cmd));
    }
  }
}

void bbts::command_runner_t::local_apply_command_runner() {

  while (true) {

    // get the command
    auto cmd = _rs->get_next_apply_command();
    if (cmd == nullptr) {
      break;
    }

    // return me that matcher for the operation
    auto ud = _udm->get_fn_impl(cmd->fun_id);

    // setup the inputs
    std::vector<tid_t> inputs; inputs.reserve(cmd->get_num_inputs());
    for (size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {
      inputs.push_back(cmd->get_input(idx).tid);
    }

    // reserve the outputs
    std::vector<std::tuple<tid_t, size_t>> outputs; outputs.reserve(cmd->get_num_outputs());

    // There are two transactions happening here. In the first transaction, the reservation
    // station grabs the input tensors and computes the outputs that will be necessary.
    // In the second transaction, the reservation station grabs the input tensors again
    // as well as the output tensors; then it runs the actual computation into the output
    // tensors.

    // calculate the output size
    _ts->local_transaction(inputs, {}, [&](const storage_t::reservation_result_t &res) {

      // make the input meta arguments
      ud_impl_t::meta_args_t input_meta_args(cmd->get_num_inputs());
      for (size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

        // store it
        auto t = res.get[idx].get().tensor;
        assert(t != nullptr);
        input_meta_args.set(idx, t->_meta);
      }

      // figure out the output arguments
      std::vector<tensor_meta_t> arguments(cmd->get_num_outputs());
      ud_impl_t::meta_args_t output_meta_args(arguments);

      // get the output meta
      ud->get_out_meta({ ._params = cmd->get_parameters() }, input_meta_args, output_meta_args);

      // setup the output arguments
      ud_impl_t::tensor_args_t output_args(cmd->get_num_outputs());
      for (size_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {
        // get the type of the output
        auto &type = ud->outputTypes[idx];
        output_meta_args.get_by_idx(idx).fmt_id = _tf->get_tensor_ftm(type);

        // the size of the tensor, tid and whether it is on the GPU
        auto ts_size = _tf->get_tensor_size(output_meta_args.get_by_idx(idx));
        auto tid = cmd->get_output(idx).tid;

        // store the outputs
        outputs.push_back({tid, ts_size});
      }
    });

    // log what is happening
    _logger->message("APPLY " + std::to_string(cmd->id) + " on my_node : " + std::to_string(_comm->get_rank()) + " Executed...\n");

    // create the outputs and run the ud
    _ts->local_transaction(inputs, outputs, [&](const storage_t::reservation_result_t &res) {

      // make the input arguments
      ud_impl_t::tensor_args_t input_args(cmd->get_num_inputs());
      for (size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

        // store it
        auto t = res.get[idx].get().tensor;
        input_args.set(idx, *t);
      }

      // setup the output arguments
      ud_impl_t::tensor_args_t output_args(cmd->get_num_outputs());
      for (size_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {

        auto t = res.create_or_get[idx].get().tensor;

        // get the type of the output
        auto &type = ud->outputTypes[idx];
        t->_meta.fmt_id = _tf->get_tensor_ftm(type);

        // set the output arg
        output_args.set(idx, *t);
      }

      // apply the ud function
      ud->call_ud(
        bbts::ud_impl_t::tensor_params_t{._params = cmd->get_parameters() },
        input_args,
        output_args);
    });

    // retire the command so it knows that we have processed the tensors
    _rs->retire_command(std::move(cmd));
  }
}


void bbts::command_runner_t::local_reduce_command_runner() {

  while (true) {

    // get the command
    auto cmd = _rs->get_next_reduce_command();
    if (cmd == nullptr) {
      break;
    }

    // check if the reduce is remote or local
    if (cmd->is_local_reduce(_comm->get_rank())) {

      // return me that matcher for matrix addition
      auto ud = _udm->get_fn_impl(cmd->fun_id);

      // preallocate the input pointers
      auto cmd_inputs = cmd->get_inputs();
      std::vector<tid_t> inputs;
      inputs.reserve(cmd_inputs.size());

      // get all the tensors we need
      for(const auto& in : cmd_inputs) {

        // get the source tensor
        inputs.push_back(in.tid);
      }

      // create the reduce op
      ud_impl_t::tensor_params_t _params = { ._params = cmd->get_parameters() };
      local_reduce_op_t op(*_tf, *_ts, inputs, _params,
                            cmd->get_output(0).tid, *ud);

      // do the apply
      op.apply();

      _logger->message("LOCAL_REDUCE " + std::to_string(cmd->id) + " on node " + std::to_string(_comm->get_rank()) + '\n');

      // retire the command so it knows that we have processed the tensors
      _rs->retire_command(std::move(cmd));


    } else {

      _logger->message("REMOTE_REDUCE_SCHEDULED");

      // forward the command to the right nodes
      if(!_comm->op_request(cmd)) {
        throw std::runtime_error("Failed to forward reduce command.");
      }

      // get the nodes involved
      auto nodes = cmd->get_nodes();

      // return me that matcher for matrix addition
      auto ud = _udm->get_fn_impl(cmd->fun_id);

      // preallocate the input pointers
      auto cmd_inputs = cmd->get_inputs();
      std::vector<tid_t> inputs;
      inputs.reserve(cmd_inputs.size());

      // get all the tensors we need
      for(const auto& in : cmd_inputs) {

        // check if the input is here
        if(in.node == _comm->get_rank()) {

          // get the source tensor
          inputs.push_back(in.tid);
        }
      }

      // make sure this does not happen
      assert(inputs.size() != 0);

      // create the move operation
      ud_impl_t::tensor_params_t _params = { ._params = cmd->get_parameters() };
      reduce_op_t op(*_comm, *_tf, *_ts, nodes, cmd->id, inputs, _params, cmd->get_output(0).tid, *ud);

      // do the apply
      op.apply();

      // retire the command so it knows that we have processed the tensors
      _rs->retire_command(std::move(cmd));

      _logger->message("REMOTE_REDUCE PROCESSED on node " + std::to_string(_comm->get_rank()) + '\n');
    }
  }
}

void bbts::command_runner_t::remote_command_handler() {

  // while we
  while (true) {

    // get the remote command
    auto cmd = _comm->expect_op_request();

    // if this is a shutdown just finish immediately
    if(cmd->type == bbts::command_t::MOVE) {

      // check if this is the move
      if(cmd->is_move()) {

        // kick off a thread to process the request
        std::thread child = std::thread([this, c = std::move(cmd)]() mutable {

          if(c->nfo.num_bytes == 0) {
            std::cout << "Empty tensor " << '\n' << std::flush;
          }

          // create the move operation
          move_op_t op(*_comm, c->id, c->nfo.num_bytes, c->get_input(0).tid, false, *_ts, c->get_input(0).node);

          // do the apply
          op.apply();

          // retire the command
          _rs->retire_command(std::move(c));
        });

        // detach the thread
        child.detach();
      }
      // check if this is a broadcast
      else {

        // kick off a thread to process the request
        std::thread child = std::thread([this, c = std::move(cmd)]() mutable {

          // get the nodes involved
          auto nodes = c->get_nodes();

          // create the move operation
          broadcast_op_t op(*_comm, *_ts, nodes, c->id, c->nfo.num_bytes, c->get_input(0).tid);

          // do the apply
          op.apply();

          // retire the command so it knows that we have processed the tensors
          _rs->retire_command(std::move(c));
        });

        // detach the thread
        child.detach();
      }
    }
    else if(cmd->type == bbts::command_t::REDUCE) {

      // kick off a thread to process the request
      std::thread child = std::thread([this, c = std::move(cmd)]() mutable {

        // get the nodes involved
        auto nodes = c->get_nodes();

        // return me that matcher for matrix addition
        auto ud = _udm->get_fn_impl(c->fun_id);

        // preallocate the input pointers
        auto cmd_inputs = c->get_inputs();
        std::vector<bbts::tid_t> inputs;
        inputs.reserve(cmd_inputs.size());

        // get all the tensors we need
        assert(!cmd_inputs.empty());
        for(const auto& in : cmd_inputs) {

          // check if the node
          if(in.node == _comm->get_rank()) {

            // get the source tensor
            inputs.push_back(in.tid);
          }
        }

        // make sure this does not happen
        assert(inputs.size() != 0);

        // create the move operation
        ud_impl_t::tensor_params_t _params = { ._params = c->get_parameters() };
        reduce_op_t op(*_comm, *_tf, *_ts, nodes, c->id, inputs, _params, c->get_output(0).tid, *ud);

        // do the apply
        op.apply();

        // retire the command
        _rs->retire_command(std::move(c));
      });

      // detach the thread
      child.detach();
    }
    else if(cmd->type == bbts::command_t::SHUTDOWN) {
      break;
    }
    else {

      // throw the runtime error
      throw std::runtime_error("This is bad can not process a command of type " + std::to_string(cmd->type));
    }
  }
}

void bbts::command_runner_t::run_deleter() {

  // while we have something remove
  tid_t id;
  while (true) {

    // get the next tensor to remove
    id = _rs->get_to_remove();
    if (id == -1) {
      break;
    }

    // deleted
    _ts->remove_by_tid(id);
    _logger->message("Remove tensor : " + std::to_string(id) + '\n');

    // remove it from the reservation station
    _rs->retire_remove(id);
  }
}

void bbts::command_runner_t::shutdown() {
  _comm->shutdown_op_request();
}

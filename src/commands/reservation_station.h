#pragma once

#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include <memory>
#include <deque>
#include "command.h"
#include "../tensor/tensor.h"
#include "../storage/storage.h"

namespace bbts {

// here we queue all the commands this node needs to execute
// and we notify all the reservation stations on the other nodes when tensors
// are ready
class reservation_station_t {
 public:

  reservation_station_t(node_id_t _node_id, int32_t num_nodes);

  // queue a command, this command has to be executed in the same thread and the commands
  // have to be queued in the exact order they are coming in
  bool queue_command(command_ptr_t _command);

  // mark that a command is processed
  bool retire_command(command_ptr_t _command);

  // Extract a list of tensors to notifiy for node. If is_done is set to true, then the station
  // has been shutdown and is done. Otherwise the returned tensors need to be notified to
  // `node`.
  [[nodiscard]] std::vector<tid_t> tensors_to_notify_node(node_id_t node, bool &is_done);

  // notify the reservation station that the tensor on an another node became available
  void notify_available_tensors(node_id_t node, const std::vector<tid_t> &tensors);

  // get the next command, you must use the result of this command as it is a unique ptr
  [[nodiscard]] command_ptr_t get_next_move_command();
  [[nodiscard]] command_ptr_t get_next_apply_command();
  [[nodiscard]] command_ptr_t get_next_reduce_command();

  // register the tensor that was added externally,
  // that is it was not created through the execution of a command
  void register_tensor(tid_t _tid);

  // returns tensors that are scheduled to be remove from the storage
  tid_t get_to_remove();

  // retire the remove command
  void retire_remove(tid_t _tid);

  // shutdown the reservation station
  void shutdown();

  // clear the reservation station
  void clear();

  // wait until all commands remote and local are executed
  void wait_until_finished();

  // execute all the scheduled commands
  void execute_scheduled_async();

  // stop executing all the commands
  void stop_executing();

  // add the hook that is triggered on scheduling
  template<class fn>
  void add_queued_hook(fn fun){ _command_queued_hook = fun; }

  // add the hook
  template<class fn>
  void add_scheduled_hook(fn fun) { _command_scheduled_hook = fun; }

  // add the retired hook
  template<class fn>
  void add_retired_hook(fn fun){ _command_retired_hook = fun; }

 private:

  // queue a remote command
  bool _queue_remote(command_ptr_t _command);

  // queue a local command
  bool _queue_local(command_ptr_t _command);

  // retire the local command
  bool _retire_command(command_ptr_t _command);

  // retire remote the command
  bool _retire_remote_command(command_ptr_t _command);

  // schedule the command for execution
  void _schedule_for_execution(command_ptr_t _cmd);

  // remove the tensor
  void _remove_tensor(tid_t _tid);

  // the state of the tensor
  struct internal_tensor_state_t {
    // the number of commands to read this, includes both the remote and local commands
    int32_t num_to_read = 0;

    // the number of commands to write this; the command is created if
    // this value equals zero
    int32_t num_to_write = -1;

    // is this tensor scheduled for delition
    bool scheduled_for_delition = false;

    bool is_created() const {
      return num_to_write == 0;
    }

    void set_created() {
      num_to_write = 0;
    }
  };

  // the mutex
  std::mutex _m;

  // we use this to wait for commands
  std::condition_variable _cv;

  // is executing
  bool _is_executing = false;

  // the number of local commands to retire
  size_t _num_local_to_retire = 0;
  size_t _num_remote_to_retire = 0;
  size_t _num_to_delete = 0;

  // a conditional variable that keeps track of how many local commands are left
  std::condition_variable _retire_cv;

  // is the node still running
  bool _shutdown = false;

  // the node for which this reservation station is for
  node_id_t _my_rank;

  // the number of nodes in the cluster
  int32_t _num_nodes;

  // the id of the last command we have executed
  command_id_t _last_cmd = -1;

  // commands ready to execute
  std::deque<command_ptr_t> _execute_ud;
  std::deque<command_ptr_t> _execute_move;
  std::deque<command_ptr_t> _execute_reduce;

  // the local commands and the number of tensors they are waiting for
  std::unordered_map<command_id_t, std::pair<command_ptr_t, int32_t>> _local_commands;

  // the local tensors commands are waiting for
  std::vector<std::unordered_multimap<tid_t, command_id_t>> _commands_waiting_for;

  // keeps all the local tensors and information about them
  std::unordered_map<tid_t, internal_tensor_state_t> _tensors;

  // keeps the status of all the remote tensors. If the tensor is present we store the command, that requested
  // the status for it. All the commands with a lower or equal command id can be executed safely as the tensor can not
  // be removed unless they are executed.
  std::vector<std::unordered_set<tid_t>> _remote_tensors;

  // tensors for which we need to send the status for, once they are created...
  std::unordered_multimap<tid_t, node_id_t> _notify_on_creation;

  // the status commands we need to send
  std::vector<std::vector<tid_t>> _send_status_queue;

  // we use this to wait for commands
  std::vector<std::condition_variable> _send_status_cv;

  // deletion cv
  std::condition_variable _deletion_cv;

  // the tensors we want to delete from storage
  std::vector<tid_t> _to_delete;

  // called when a command is retired on this node
  std::function<void(command_id_t id)> _command_retired_hook;

  // called when a command is scheduled on this node
  std::function<void(command_id_t id)> _command_scheduled_hook;

  // called when a command is scheduled on this node
  std::function<void(command_id_t id)> _command_queued_hook;

};

using reservation_station_ptr_t = std::shared_ptr<reservation_station_t>;

}

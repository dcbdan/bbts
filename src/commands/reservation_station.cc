#include "reservation_station.h"
#include "../server/static_config.h"
#include "command.h"
#include <cassert>

bbts::reservation_station_t::reservation_station_t(
  bbts::node_id_t _node_id, int32_t num_nodes) :
    _my_rank(_node_id),
    _num_nodes(num_nodes),
    _send_status_cv(num_nodes),
    _commands_waiting_for(num_nodes)
{
  // make one of these for each node
  _remote_tensors.resize(num_nodes);
  _send_status_queue.resize(num_nodes);

  // just empty hooks
  _command_retired_hook = [](command_id_t _) {};
  _command_scheduled_hook = [](command_id_t _) {};
  _command_queued_hook = [](command_id_t _) {};
}

bool bbts::reservation_station_t::queue_command(bbts::command_ptr_t _command) {

  // lock the reservation station
  std::unique_lock<std::mutex> lk(_m);

  // make sure the commands are scheduled in order
  auto cmd_id = _command->id;
  if(_last_cmd >= cmd_id) {
    return false;
  }

  // figure out whether this is a local command or a remote command
  // (a local command is initiated at this node,
  //  a remote command is initiated on another node)
  bool success;
  if(_command->get_root_node() == _my_rank) {

    // queue the local command
    success = _queue_local(std::move(_command));
  }
  else {
    success = _queue_remote(std::move(_command));
  }

  // if we succeeded we need to update the last command
  if(success) {
    _last_cmd = cmd_id;
  }

  // unlock here
  lk.unlock();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // notify that we have queued a command
    _command_queued_hook(cmd_id);
  }

  // we are done get out of here
  return success;
}

bool bbts::reservation_station_t::retire_command(bbts::command_ptr_t _command) {

  // lock the reservation station
  std::unique_lock<std::mutex> lk(_m);

  // store the command id
  auto cmd_id = _command->id;

  // check if this is a remote node
  bool success;
  if(_command->get_root_node() == _my_rank) {
    success = _retire_command(std::move(_command));
  } else {
    success = _retire_remote_command(std::move(_command));
  }

  // unlock here
  lk.unlock();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // notify that we have retired a command
    _command_retired_hook(cmd_id);
  }

  // we are done get out of here
  return success;
}

std::vector<bbts::tid_t> bbts::reservation_station_t::tensors_to_notify_node(
  bbts::node_id_t node,
  bool &is_done)
{
  std::unique_lock<std::mutex> lk(_m);

  // wait until we are shutdown or we have something to notify for this node
  _send_status_cv[node].wait(lk, [&]{
    return !_send_status_queue[node].empty() || _shutdown;
  });

  // if we have shutdown return
  if(_shutdown && _send_status_queue[node].empty()) {
    is_done = true;
    return {};
  }

  // get the tensors we need to send
  std::vector<tid_t> out;
  std::swap(out, _send_status_queue[node]);

  // return them
  is_done = false;
  return std::move(out);
}

void bbts::reservation_station_t::notify_available_tensors(
  bbts::node_id_t node,
  const std::vector<tid_t> &tensors)
{
  // lock the tensor
  std::unique_lock<std::mutex> lk(_m);

  // go through tensors
  for(auto t : tensors) {

    // go through all the commands that are waiting for this tensor
    auto cw = _commands_waiting_for[node].equal_range(t);
    for (auto it = cw.first; it != cw.second;) {

      // try to find the command
      auto jt = _local_commands.find(it->second);
      assert(jt != _local_commands.end());

      // check if we have all the inputs
      if(0 == (--jt->second.second)) {

        // schedule the command for execution
        _schedule_for_execution(std::move(jt->second.first));

        // remove the command
        _local_commands.erase(jt);
      }

      // remove the command from the waiting list
      it = _commands_waiting_for[node].erase(it);
    }

    // mark that the tensor is here
    _remote_tensors[node].insert(t);
  }
}

void bbts::reservation_station_t::register_tensor(bbts::tid_t _tid) {

  // lock the tensor
  std::unique_lock<std::mutex> lk(_m);

  // get the tensor state if any
  auto &s = _tensors[_tid];

  // make sure that it was not created before
  assert(!s.is_created());

  // we are done writing to the tensor and
  s.set_created();

  // go through the commands that are waiting
  auto cw = _commands_waiting_for[_my_rank].equal_range(_tid);
  for (auto it = cw.first; it != cw.second;) {

    // try to find the command
    auto jt = _local_commands.find(it->second);
    assert(jt != _local_commands.end());

    // check if we have all the inputs
    if(0 == (--jt->second.second)) {

      // schedule the command for execution
      _schedule_for_execution(std::move(jt->second.first));

      // remove the command
      _local_commands.erase(jt);
    }

    // remove the command from the waiting list
    it = _commands_waiting_for[_my_rank].erase(it);
  }

  // go through all nodes we need to notify once this tensor is created
  auto nd = _notify_on_creation.equal_range(_tid);
  while (nd.first != nd.second) {

    // add it to the send status queue so that we can notify the node
    auto it = nd.first;
    _send_status_queue[it->second].push_back(_tid);
    _send_status_cv[it->second].notify_all();

    // erase the ones we don't need
    _notify_on_creation.erase(++nd.first, nd.second);
  }
}

bbts::command_ptr_t bbts::reservation_station_t::get_next_move_command() {

  // wait until we have something here
  std::unique_lock<std::mutex> lk(_m);
  _cv.wait(lk, [&]{ return (!_execute_move.empty() && _is_executing)  || _shutdown ; });

  // if we have shutdown return null as we have no command left...
  if(_shutdown) {
    return nullptr;
  }

  // pop the unique pointer of the vector
  auto tmp = _execute_move.pop();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // mark that we have scheduled this command
    _command_scheduled_hook(tmp->id);
  }

  // return it
  return std::move(tmp);
}

bbts::command_ptr_t bbts::reservation_station_t::get_next_apply_command() {

  // wait until we have something here
  std::unique_lock<std::mutex> lk(_m);
  _cv.wait(lk, [&]{ return (!_execute_ud.empty() && _is_executing)  || _shutdown ; });

  // if we have shutdown return null as we have no command left...
  if(_shutdown) {
    return nullptr;
  }

  // pop the unique pointer of the vector
  auto tmp = _execute_ud.pop();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // mark that we have scheduled this command
    _command_scheduled_hook(tmp->id);
  }

  // return it
  return std::move(tmp);
}

bbts::command_ptr_t bbts::reservation_station_t::get_next_reduce_command() {

  // wait until we have something here
  std::unique_lock<std::mutex> lk(_m);
  _cv.wait(lk, [&]{ return (!_execute_reduce.empty() && _is_executing)  || _shutdown ; });

  // if we have shutdown return null as we have no command left...
  if(_shutdown) {
    return nullptr;
  }

  // pop the unique pointer of the vector
  auto tmp = _execute_reduce.pop();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // mark that we have scheduled this command
    _command_scheduled_hook(tmp->id);
  }

  // return it
  return std::move(tmp);
}

bbts::tid_t bbts::reservation_station_t::get_to_remove() {

  // wait until we have something to delete
  std::unique_lock<std::mutex> lk(_m);
  _deletion_cv.wait(lk, [&]{ return !_to_delete.empty() || _shutdown; });

  // if we have nothing to delete
  if(_to_delete.empty()) {
    return -1;
  }

  // return the tensor we want to delete
  auto tmp = _to_delete.back(); _to_delete.pop_back();
  return tmp;
}

void bbts::reservation_station_t::retire_remove(tid_t _tid) {

  // lock here
  std::unique_lock<std::mutex> lk(_m);

  // we got one less
  _num_to_delete--;
  if(_num_to_delete == 0 && _num_local_to_retire == 0 && _num_remote_to_retire == 0) {
    _retire_cv.notify_all();
  }
}

void bbts::reservation_station_t::shutdown() {

  // set the flag
  std::unique_lock<std::mutex> lk(_m);
  _shutdown = true;

  // notify that we are done
  _cv.notify_all();
  _deletion_cv.notify_all();
  for(auto &cv : _send_status_cv) { cv.notify_all(); }
}

void bbts::reservation_station_t::clear() {

  std::unique_lock<std::mutex> lk(_m);

  _last_cmd = -1;
  _execute_move.clear();
  _execute_reduce.clear();
  _execute_ud.clear();
  _local_commands.clear();

  for(auto &ls : _commands_waiting_for) {
    ls.clear();
  }
  _tensors.clear();

  for(auto &rt : _remote_tensors) {
    rt.clear();
  }
  _notify_on_creation.clear();

  for(auto &t : _send_status_queue) {
    t.clear();
  }

  _to_delete.clear();
}

void bbts::reservation_station_t::wait_until_finished() {

  // wait until all the commands are run
  std::unique_lock<std::mutex> lk(_m);
  _retire_cv.wait(lk, [&]{
    return _num_local_to_retire == 0 && _num_remote_to_retire == 0 && _num_to_delete == 0;
  });
}

void bbts::reservation_station_t::execute_scheduled_async() {

  // kick off everything
  std::unique_lock<std::mutex> lk(_m);
  _is_executing = true;
  _cv.notify_all();
}

void bbts::reservation_station_t::stop_executing() {

  // update the flag
  std::unique_lock<std::mutex> lk(_m);
  _is_executing = false;
  _last_cmd = -1;
}

bool bbts::reservation_station_t::_queue_remote(bbts::command_ptr_t _command) {

  // handle delete
  if(_command->is_delete()) {
    // this is bad a remote delete should never exist... // TODO handle this gracefully
    return false;
  }

  // get the node that is going to initiate the execution of this command
  auto rootNode = _command->get_root_node();

  // count the number of inputs that are not present
  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto _in = _command->get_input(i);

    // check if this input is supposed to be located on this machine
    if(_in.node == _my_rank) {

      // if it is we need to mark that it is being read and
      auto &s = _tensors[_in.tid];

      // mark that we are reading this tensor
      s.num_to_read++;

      // check if it was not created we need to mark that
      // we need to notify the remote node once the tensor is created
      if(!s.is_created()) {

        // mark that we need to notify once this tid is created _in.node
        _notify_on_creation.insert({_in.tid, rootNode});
      }
      else {

        // add it to the send status queue so that we can notify the node
        _send_status_queue[rootNode].push_back(_in.tid);
        _send_status_cv[rootNode].notify_all();
      }
    }
  }

  // go through the output tensors
  for(auto idx = 0; idx < _command->get_num_outputs(); ++idx) {

    // grab the output tensor
    auto &_out = _command->get_output(idx);

    // check if the node
    if(_out.node == _my_rank) {

      // get the tid
      auto &s = _tensors[_out.tid];

      // make sure everything is fine TODO I need to recover from this somehow...
      if(s.scheduled_for_delition && s.is_created()) { return false; }

      // we are the only writer to this tensor
      // (the assumption is that (1) this can't be a touch command and
      //  (2) all commands that are not touch need to be written to once)
      assert(!_command->is_touch());
      s.num_to_write = 1;
    }
  }

  // we got another remote command
  _num_remote_to_retire++;

  return true;
}

bool bbts::reservation_station_t::_queue_local(bbts::command_ptr_t _command) {

  // handle delete
  if(_command->is_delete()) {

    // go through the inputs and eiter remove the tensors directly,
    // or mark them for deletion if they are going to be used soon
    for(auto idx = 0; idx < _command->get_num_inputs(); ++idx) {

      // grab the input tensor
      auto &in = _command->get_input(idx);

      // mark the tensor as scheduled for deletion
      auto &s = _tensors[in.tid];

      // if we created the tensor, if not just delete it!
      if(s.num_to_read == 0 && s.num_to_write == 0) {

        // remove the tensor immediately
        _remove_tensor(in.tid);
      }
      else {

        // ok the tensor is not ready for deletion schedule it
        assert(s.scheduled_for_delition == false);
        s.scheduled_for_delition = true;
      }

      // we got one more to delete
      _num_to_delete++;
    }

    // finish delete processed
    return true;
  }

  // count the number of inputs that are not present
  int32_t num_not_present = 0;
  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto _in = _command->get_input(i);

    // check if this is a remote tensor or a local tensor
    if(_in.node == _my_rank) {

      // if it is a local we need to check if it was created already
      auto &s = _tensors[_in.tid];

      // make sure that this tensor was not deleted before this TODO I need to recover from this somehow...
      if(s.scheduled_for_delition) { return false; }

      // we are reading this tensor
      s.num_to_read++;

      // if it was not created we need to keep track of that
      if(!s.is_created()) {

        // tensor is not present
        num_not_present++;

        // mark that this command is waiting
        _commands_waiting_for[_my_rank].insert({_in.tid, _command->id});
      }
    }
    else {

      // ok this is a remote tensor, we need to check if it is present
      auto &_rts = _remote_tensors[_in.node];
      auto _ts = _rts.find(_in.tid);

      // if the tensor is not present we have to mark that we are waiting for
      // a notification from a remote node
      if(_ts == _rts.end()) {

        // tensor is not present
        num_not_present++;

        // mark the this command is waiting for a remote tensor
        _commands_waiting_for[_in.node].insert({_in.tid, _command->id});
      }
    }
  }

  // go through the output tensors
  for(auto idx = 0; idx < _command->get_num_outputs(); ++idx) {

    // grab the output tensor
    auto &_out = _command->get_output(idx);

    // check if the node
    if(_out.node == _my_rank) {

      // get the tid
      auto &s = _tensors[_out.tid];

      // make sure everything is fine TODO I need to recover from this somehow...
      if(s.scheduled_for_delition && s.is_created()) { return false; }

      // we are writing to this tensor
      if(_command->is_touch()) {
        // if this is the first touch being queued,
        // set the num_to_write value for the tensor
        if(s.num_to_write == -1) {
          s.num_to_write = _command->nfo.num_writes;
        }
      } else {
        assert(s.num_to_write == -1);
        s.num_to_write = 1;
      }
    }
  }

  // if we have all the required tensors we can kick off the command
  if(num_not_present == 0) {

    switch (_command->type) {
      case command_t::op_type_t::APPLY:  _execute_ud.push    (std::move(_command)); break;
      case command_t::op_type_t::TOUCH:  _execute_ud.push    (std::move(_command)); break;
      case command_t::op_type_t::MOVE:   _execute_move.push  (std::move(_command)); break;
      case command_t::op_type_t::REDUCE: _execute_reduce.push(std::move(_command)); break;
      default: assert(false); // this is not supposed to happen
    }

    _cv.notify_all();
  }
  else {

    // store the number of tensors this command is waiting for
    auto cmd_id = _command->id;
    _local_commands[cmd_id] = { std::move(_command),  num_not_present };
  }

  // we have more local command that we need to retire
  _num_local_to_retire++;

  // we are done here
  return true;
}

bool bbts::reservation_station_t::_retire_command(bbts::command_ptr_t _command) {

  // if this is a delete we remove the tensor
  if (_command->type == command_t::op_type_t::DELETE) {

    // remove the tensors
    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto t = _command->get_input(i);

      // remove the tensor immediately
      _remove_tensor(t.tid);

      // remove the command
      _local_commands.erase(_command->id);
    }

    return true;
  }

  // make sure to go through the written to tensors
  for(int32_t i = 0; i < _command->get_num_outputs(); i++) {

    // get the tensor required in the output
    auto out = _command->get_output(i);

    // if this tensor is not on our node we don't do anything
    if(out.node != _my_rank) {
      continue;
    }

    // get the tid
    auto tid = out.tid;
    auto &s = _tensors[tid];

    // make sure that it was not created before and that it is expecting writes
    assert(s.num_to_write > 0);

    // we have written to the tensor once
    s.num_to_write--;

    // if the tensor is waiting for more writes, there is nothing else to do
    if(s.num_to_write != 0) {
      continue;
    }
    // At this point, the tensor has been completely written to and is fully
    // created.

    // remove the tensor if has been written to completely and is not needed
    if (s.num_to_read == 0 && s.scheduled_for_delition) {

      // remove the tensor immediately
      _remove_tensor(out.tid);
    }

    // go through the commands that are waiting
    auto cw = _commands_waiting_for[_my_rank].equal_range(tid);
    for (auto it = cw.first; it != cw.second;) {

      // try to find the command
      auto jt = _local_commands.find(it->second);
      assert(jt != _local_commands.end());

      // check if we have all the inputs
      if (0 == (--jt->second.second)) {

        // schedule the command for execution
        _schedule_for_execution(std::move(jt->second.first));

        // remove the command
        _local_commands.erase(jt);
      }

      // remove the command from the waiting list
      it = _commands_waiting_for[_my_rank].erase(it);
    }

    // go through all nodes we need to notify once this tensor is created
    auto nd = _notify_on_creation.equal_range(tid);
    while(nd.first != nd.second) {

      // add it to the send status queue so that we can notify the node
      auto it = nd.first;
      _send_status_queue[it->second].push_back(tid);
      _send_status_cv[it->second].notify_all();

      // erase the ones we don't need
      _notify_on_creation.erase(++nd.first, nd.second);
    }
  }

  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto in = _command->get_input(i);

    // if this tensor is not on our node we don't do anything
    if(in.node != _my_rank) {
      continue;
    }

    // get the tid
    auto tid = in.tid;
    auto &s = _tensors[tid];

    // decrement the number of readers
    s.num_to_read--;
    assert(s.num_to_read >= 0);

    // if there are no command that is writing to this tensor,
    // reading this tensor and the tensor is scheduled for deletion, delete it here
    if (s.num_to_read == 0 && s.num_to_write == 0 && s.scheduled_for_delition) {

      // remove the tensor immediately
      _remove_tensor(in.tid);
    }
  }

  // remove the command
  _local_commands.erase(_command->id);

  // we have one less to retire
  _num_local_to_retire--;
  if(_num_to_delete == 0 && _num_local_to_retire == 0 && _num_remote_to_retire == 0) {
    _retire_cv.notify_all();
  }

  return true;
}
bool bbts::reservation_station_t::_retire_remote_command(bbts::command_ptr_t _command) {

  // if this is a delete we remove the tensor
  if (_command->type == command_t::op_type_t::DELETE) {
    throw std::runtime_error("There are no remote delete instructions...");
  }

  // make sure to go through the created tensors
  for(int32_t i = 0; i < _command->get_num_outputs(); i++) {

    // get the tensor required in the output
    auto out = _command->get_output(i);

    // if this tensor is not on our node we don't do anything
    if(out.node != _my_rank) {
      continue;
    }

    // get the tid
    auto tid = out.tid;
    auto &s = _tensors[tid];

    if(s.is_created()) {
      std::stringstream ss;
      _command->print(ss);
      std::cout << ss.str();
      std::cout << s.num_to_read << ", " << s.num_to_write << std::endl;
    }
    assert(!s.is_created());
    s.num_to_write--;

    if(s.num_to_write != 0) {
      continue;
    }

    // remove the tensor if it is not needed
    if (s.num_to_read == 0 && s.scheduled_for_delition) {

      // remove the tensor immediately
      _remove_tensor(out.tid);
    }

    // go through the commands that are waiting
    auto cw = _commands_waiting_for[_my_rank].equal_range(tid);
    for (auto it = cw.first; it != cw.second;) {

      // try to find the command
      auto jt = _local_commands.find(it->second);
      assert(jt != _local_commands.end());

      // check if we have all the inputs
      if (0 == (--jt->second.second)) {

        // schedule the command for execution
        _schedule_for_execution(std::move(jt->second.first));

        // remove the command
        _local_commands.erase(jt);
      }

      // remove the command from the waiting list
      it = _commands_waiting_for[_my_rank].erase(it);
    }

    // go through all nodes we need to notify once this tensor is created
    auto nd = _notify_on_creation.equal_range(tid);
    while(nd.first != nd.second) {

      // add it to the send status queue so that we can notify the node
      auto it = nd.first;
      _send_status_queue[it->second].push_back(tid);
      _send_status_cv[it->second].notify_all();

      // erase the ones we don't need
      _notify_on_creation.erase(++nd.first, nd.second);
    }
  }

  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto in = _command->get_input(i);

    // if this tensor is not on our node we don't do anything
    if(in.node != _my_rank) {
      continue;
    }

    // get the tid
    auto tid = in.tid;
    auto &s = _tensors[tid];

    // decrement the number of readers
    s.num_to_read--;
    assert(s.num_to_read >= 0);

    // if there are no command that is writing to this tensor
    // reading this tensor and the tensor is scheduled for deletion, delete it here
    if (s.num_to_read == 0 && s.num_to_write == 0 && s.scheduled_for_delition) {

      // remove the tensor immediately
      _remove_tensor(in.tid);
    }
  }

  // remote command retired
  _num_remote_to_retire--;
  if(_num_to_delete == 0 && _num_local_to_retire == 0 && _num_remote_to_retire == 0) {
    _retire_cv.notify_all();
  }

  return true;
}

void bbts::reservation_station_t::_schedule_for_execution(bbts::command_ptr_t _cmd) {

  // schedule the command for execution
  switch (_cmd->type) {
    case command_t::op_type_t::APPLY:  _execute_ud.push(    std::move(_cmd)); break;
    case command_t::op_type_t::TOUCH:  _execute_ud.push(    std::move(_cmd)); break;
    case command_t::op_type_t::MOVE:   _execute_move.push(  std::move(_cmd)); break;
    case command_t::op_type_t::REDUCE: _execute_reduce.push(std::move(_cmd)); break;
    default: assert(false); // this is not supposed to happen
  }

  _cv.notify_all();
}

void bbts::reservation_station_t::_remove_tensor(bbts::tid_t _tid) {

  // remove the tensor from the storage
  _to_delete.push_back(_tid);

  // remove the tensor
  _tensors.erase(_tid);

  // make sure there are not commands waiting for the delete
  assert(_commands_waiting_for[_my_rank].find(_tid) == _commands_waiting_for[_my_rank].end());

  // notify that there is stuff to delete
  _deletion_cv.notify_all();
}

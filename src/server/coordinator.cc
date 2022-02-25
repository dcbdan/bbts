#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include "coordinator.h"
#include "../utils/terminal_color.h"
#include "node_config.h"

using namespace std::chrono;

bbts::coordinator_t::coordinator_t(bbts::communicator_ptr_t _comm,
                                   bbts::gpu_scheduler_ptr_t _gpu_scheduler,
                                   bbts::reservation_station_ptr_t _rs,
                                   bbts::logger_ptr_t _logger,
                                   storage_ptr_t _storage,
                                   bbts::command_runner_ptr_t _command_runner,
                                   bbts::tensor_notifier_ptr_t _tensor_notifier,
                                   bbts::udf_manager_ptr _udf_manager,
                                   bbts::tensor_factory_ptr_t _tf)

    : _comm(std::move(_comm)),
      _gpu_scheduler(std::move(_gpu_scheduler)),
      _rs(std::move(_rs)),
      _logger(std::move(_logger)),
      _storage(std::move(_storage)),
      _command_runner(std::move(_command_runner)),
      _tensor_notifier(std::move(_tensor_notifier)),
      _udf_manager(std::move(_udf_manager)),
      _tf(std::move(_tf)) { _is_down = false; }

void bbts::coordinator_t::accept() {

  while (!_is_down) {

    // the operation
    auto op = _comm->expect_coord_op();

    std::stringstream ss;
    switch (op._type) {

      case coordinator_op_types_t::FAIL : {
        _fail();
        break;
      }
      case coordinator_op_types_t::RUN : {
        _run();
        break;
      }
      case coordinator_op_types_t::CLEAR : {
        _clear();
        break;
      }
      case coordinator_op_types_t::SCHEDULE : {
        _schedule(op, ss);
        break;
      }
      case coordinator_op_types_t::SHUTDOWN : {
        _shutdown();
        break;
      }
      case coordinator_op_types_t::VERBOSE : {
        _set_verbose(static_cast<bool>(op._val));
        break;
      }
      case coordinator_op_types_t::MAX_STORAGE : {
        _set_max_storage(op._val);
        break;
      }
      case coordinator_op_types_t::PRINT_STORAGE : {
        _print_storage(ss);
        break;
      }
      case coordinator_op_types_t::PRINT_TENSOR : {
        _print_tensor((tid_t)(op._val), ss);
        break;
      }
      case coordinator_op_types_t::REGISTER : {
        _register(op, ss);
        break;
      }
      case coordinator_op_types_t::FETCH_META : {
        _handle_fetch_meta(ss);
        break;
      }
    }

    // sync all nodes
    _comm->send_response_string(ss.str());
  }
}

std::tuple<bool, std::string> bbts::coordinator_t::schedule_commands(const std::vector<command_ptr_t> &cmds) {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::SCHEDULE,
      ._val = cmds.size()})) {
    return {false, "Could not schedule commands!\n"};
  }

  // send all the commands
  if (!_comm->send_coord_cmds(cmds)) {
    return {false, "Could not send the commands we were about to schedule!\n"};
  }

  // load all the commands
  std::stringstream ss;
  _load_cmds(cmds, ss);

  // collect the respnses from all the nodes
  std::tuple<bool, std::string> out = {true, ""};
  _collect(out);

  // check if we succeded
  if(!std::get<0>(out) || !std::get<1>(out).empty()) {
    return {false, std::get<1>(out).empty() ? "Unknown error\n" : std::get<1>(out)};
  }

  // we succeded
  return {true, "Scheduled " + std::to_string(cmds.size()) + " commands\n"};
}

std::tuple<bool, std::string> bbts::coordinator_t::compile_commands(float gpu_transfer_cost_per_byte,
                                                                    float send_cost_per_byte,
                                                                    const std::vector<abstract_command_t> &cmds,
                                                                    const std::vector<abstract_ud_spec_t> &funs) {

  // fetch the info about the tensors
  std::unordered_map<bbts::tid_t, bbts::tensor_meta_t> meta;
  std::vector<std::unordered_set<bbts::tid_t>> locations;
  _fetch_tensor_info(meta, locations);

  try {

    // make the cost
    cost_model_ptr_t cost = std::make_shared<cost_model_t>(meta,
                                                          funs,
                                                          _tf,
                                                          _udf_manager,
                                                          gpu_transfer_cost_per_byte,
                                                          send_cost_per_byte);

    // init the compiler
    two_layer_compiler compiler(cost, _comm->get_num_nodes());

    // the compiled commands
    auto compiled_cmds = compiler.compile(cmds, locations);

    std::stringstream ss2;
    for(auto &c : compiled_cmds) {
      c->print(ss2);
    }
    //std::cout << ss2.str() << '\n';

    // schedule the compiled commands
    return schedule_commands(compiled_cmds);
  }
  catch (const std::runtime_error& ex) {
    return {false, ex.what()};
  }
}

std::tuple<bool, std::string> bbts::coordinator_t::run_commands() {

  // measure start
  auto start = high_resolution_clock::now();

  // send the commands
  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::RUN, ._val = 0})) {
    return {false, "Could not run the commands!\n"};
  }

  // run everything
  _run();

  // collect the responses from all the nodes
  std::tuple<bool, std::string> out = {true, ""};
  _collect(out);

  // measure end
  auto end = high_resolution_clock::now();
  auto duration = (double) duration_cast<microseconds>(end - start).count() / (double) duration_cast<microseconds>(1s).count();

  return {true, "Finished running commands in " + std::to_string(duration) + "s \n"};
}

std::tuple<bool, std::string> bbts::coordinator_t::set_verbose(bool val) {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::VERBOSE,
      ._val = static_cast<size_t>(val)})) {
    return {false, "Failed to set the verbose flag!\n"};
  }

  // run everything
  _set_verbose(val);

  // collect the responses from all the nodes
  std::tuple<bool, std::string> out = {true, "Set the verbose flag to " + std::to_string(val) + "\n"};
  _collect(out);

  return out;
}

std::tuple<bool, std::string> bbts::coordinator_t::set_num_threads(std::uint32_t set_num_threads) {

  // TODO - need some work
  return {false, "Not supported for now!"};
}

std::tuple<bool, std::string> bbts::coordinator_t::set_max_storage(size_t val) {

  // send the command
  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::MAX_STORAGE,
                                             ._val = val})) {

    return {false, "Failed to set the maximum storage flag!\n"};
  }

  // run everything
  _set_max_storage(val);

  // collect the responses from all the nodes
  std::tuple<bool, std::string> out = {true, "Set the max storage to " + std::to_string(val) + " bytes\n"};
  _collect(out);

  return out;
}

std::tuple<bool, std::string> bbts::coordinator_t::print_storage_info() {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::PRINT_STORAGE, ._val = 0})) {
    return {false, "Failed to print storage!\n"};
  }

  // print the storage
  std::stringstream ss;
  _print_storage(ss);

  // collect the responses from all the nodes
  std::tuple<bool, std::string> out = {true, ss.str()};
  _collect(out);

  // we succeded
  return out;
}

std::tuple<bool, std::string> bbts::coordinator_t::print_tensor_info(bbts::tid_t id) {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::PRINT_TENSOR, ._val = (size_t)(id) } )) {
    return {false, "Failed to print tensor!\n"};
  }

  // print the storage
  std::stringstream ss;
  _print_tensor(id, ss);

  // sync everything
  std::tuple<bool, std::string> out = {true, ss.str()};
  _collect(out);

  // we succeded
  return out;
}

std::tuple<bool, std::string> bbts::coordinator_t::clear() {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::CLEAR, ._val = 0})) {
    return {false, "Failed to clear the cluster!\n"};
  }

  // claer the storage
  _clear();

  // sync everything
  std::tuple<bool, std::string> out = {true, "Cleared the cluster!\n"};
  _collect(out);

  // we succeded
  return out;
}

std::tuple<bool, std::string> bbts::coordinator_t::shutdown_cluster() {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::SHUTDOWN, ._val = 0})) {
    return {false, "Failed to shutdown the cluster!\n"};
  }

  // print the storage
  _shutdown();

  // sync everything
  std::tuple<bool, std::string> out = {true, "Cluster shutdown!\n"};
  _collect(out);

  // we succeded
  return out;
}

std::tuple<bool, std::string> bbts::coordinator_t::_fetch_tensor_info(std::unordered_map<bbts::tid_t, bbts::tensor_meta_t> &meta,
                                                                      std::vector<std::unordered_set<bbts::tid_t>> &locations) {

  // send the request to get all the meta
  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::FETCH_META, ._val = 0})) {
    return {false, "Failed to shutdown the cluster!\n"};
  }

  // init the locations
  locations.clear(); locations.resize(_comm->get_num_nodes());

  // get the meta from my storage
  auto m = _storage->extract_meta();
  for(auto &t : m) {
    meta[std::get<0>(t)] = std::get<1>(t);
    locations[0].insert(std::get<0>(t));
  }

  // recive all the meta from other nodes
  bool success = true;
  for(node_id_t node = 1; node < _comm->get_num_nodes(); ++node) {

    // fetch the meta
    if(!_comm->recv_meta(node, m)) {
      success = false;
    }
    else {

      // store the meta
      for(auto &t : m) {
        meta[std::get<0>(t)] = std::get<1>(t);
        locations[node].insert(std::get<0>(t));
      }
    }
  }

    // sync everything
  std::tuple<bool, std::string> out = {success, "Fetched meta data \n"};
  _collect(out);

  return out;
}

void bbts::coordinator_t::_fail() {
  std::cout << bbts::red << "FAIL\n" << bbts::reset;
  exit(-1);
}

std::tuple<bool, std::string> bbts::coordinator_t::load_shared_library(char* file_bytes, size_t file_size) {

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::REGISTER, ._val = file_size})) {
    return {false, "Failed to register library!\n"};
  }

  // send the data
  if (!_comm->send_bytes(file_bytes, file_size)) {
    return {false, "Could not send file to register!\n"};
  }

  // do the actual registering, on this node
  std::stringstream ss;
  bool val = _register_from_bytes(file_bytes, file_size, ss);

  // sync everything
  std::tuple<bool, std::string> out;
  if(!val) {
    out = {val, ss.str()};
  }
  else {
    out = {val, "Loaded successfully!\n"};
  }
  _collect(out);

  // return the output
  return out;
}

void bbts::coordinator_t::_schedule(coordinator_op_t op, std::stringstream &ss) {

  // expect all the commands
  std::vector<command_ptr_t> cmds;
  if (!_comm->expect_coord_cmds(op._val, cmds)) {
    std::cout << bbts::red << "Could not receive the scheduled commands!\n" << bbts::reset;
    return;
  }

  // load all the commands
  _load_cmds(cmds, ss);
}

void bbts::coordinator_t::_collect(std::tuple<bool, std::string> &out) {

  // collect all the responses
  for(bbts::tid_t node = 1; node < _comm->get_num_nodes(); ++node) {
    auto rec = _comm->expect_response_string(node);

    // combine the result
    std::get<0>(out) = std::get<0>(out) &&  std::get<0>(rec);
    std::get<1>(out) = std::get<1>(out)  +  std::get<1>(rec);
  }
}

void bbts::coordinator_t::_load_cmds(const std::vector<command_ptr_t> &cmds,
                                     std::stringstream &ss) {

  // schedule them all at once
  for (auto &_cmd : cmds) {

    // if it uses the node
    if (_cmd->uses_node(_comm->get_rank())) {
      _rs->queue_command(_cmd->clone());
    }
  }
}

void bbts::coordinator_t::_run() {

  // async execute the scheduled commands
  _rs->execute_scheduled_async();

  // wait for all the commands to be run
  _rs->wait_until_finished();

  // stop executing all the commands
  _rs->stop_executing();
}

void bbts::coordinator_t::_clear() {

  // clear everything
  _storage->clear();
  _rs->clear();
}

void bbts::coordinator_t::_set_verbose(bool val) {
  _logger->set_enabled(val);
}

void bbts::coordinator_t::_print_storage(std::stringstream &ss) {

  ss << "<<< For Node " << _comm->get_rank() << ">>>\n";
  _storage->print(ss);
}

void bbts::coordinator_t::_print_tensor(tid_t id, std::stringstream &ss) {

  // check if it exists
  if(!_storage->has_tensor(id)) {
    return;;
  }

  // run the transaction
  _storage->local_transaction({id}, {}, [&](const storage_t::reservation_result_t &res) {

    // the get the tensor
    auto ts = res.get[0].get().tensor;
    if(ts != nullptr) {

      // print the tensor since we found it
      ss << bbts::green << "<<< On Node " << _comm->get_rank() << ">>>\n" << bbts::reset;
      _tf->print_tensor(ts, ss);
    }
  });
}

bool bbts::coordinator_t::_register(coordinator_op_t op, std::stringstream &ss) {

  std::vector<char> file_bytes;
  file_bytes.reserve(op._val);

  if(!_comm->expect_bytes(op._val, file_bytes)) {
    ss << bbts::red << "Could not recieve the library file!\n" << bbts::reset;
    return false;
  }

  return _register_from_bytes(file_bytes.data(), op._val, ss);
}

void bbts::coordinator_t::_handle_fetch_meta(std::stringstream &ss) {

  auto m = _storage->extract_meta();
  if(_comm->send_tensor_meta(m)) {
    ss << bbts::red << "Failed to send the tensor meta data." << bbts::reset;
  }
}

bool bbts::coordinator_t::_register_from_bytes(char* file_bytes, size_t file_size, std::stringstream &ss) {

  // make the temporary file name
  int rank = _comm->get_rank();
  std::string filename = std::string("/tmp/bbts_lib_") + std::to_string(_comm->get_rank())  + "_" + std::to_string(shared_library_item_t::last_so) + ".so";

  // this will modify filename
  int filedes = open(filename.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0777);

  // check if we could actually open this
  if(filedes == -1) {
    ss << bbts::red << "Could not set temporary filename!\n" << bbts::reset;
    return false;
  }
  if(-1 == write(filedes, file_bytes, file_size)) {
    ss << bbts::red << "Could not write shared library object!\n" << bbts::reset;
    return false;
  }

  // close the file
  close(filedes);

  // open the newly created temporary file
  void* so_handle = dlopen(filename.c_str(), RTLD_LOCAL | RTLD_NOW);
  if(!so_handle) {
    ss << bbts::red << "Could not open temporary shared library object " << dlerror() << "!\n" << bbts::reset;
    return false;
  }

  // The .so should have atleast one of the two (unmangled) functions, register_tensors, register_udfs.
  bool had_something = false;
  void* register_tensors_ = dlsym(so_handle, "register_tensors");
  if(register_tensors_) {
    had_something = true;
    typedef void *register_tensors_f(tensor_factory_ptr_t);
    auto *register_tensors = (register_tensors_f *) register_tensors_;
    register_tensors(_tf);
  }

  // check for the register_udfs
  void* register_udfs_ = dlsym(so_handle, "register_udfs");
  if(register_udfs_) {
    had_something = true;
    typedef void *register_udfs_f(udf_manager_ptr);
    auto *register_udfs = (register_udfs_f *) register_udfs_;
    register_udfs(_udf_manager);
  }

  // check if we have actually loaded something
  if(!had_something) {
    ss << bbts::red << "Shared library object did not have a valid \"register_tensors\" or \"register_udfs\"!\n" << std::endl;
  }

  // keep track of the stuff here so the system can clean it up later
  shared_libs.emplace_back(filename, so_handle);

  return had_something;
}

// we start counting so libararies naturally from zero
int64_t bbts::coordinator_t::shared_library_item_t::last_so = 1;

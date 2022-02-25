#include <bits/stdint-intn.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include <filesystem>
#include "../../src/tensor/tensor.h"
#include "../../src/tensor/tensor_factory.h"
#include "../../src/server/node.h"
#include "../../src/utils/terminal_color.h"

#include "../../third_party/cli/include/cli/cli.h"
#include "../../third_party/cli/include/cli/clifilesession.h"

using namespace cli;

std::thread loading_message(std::ostream &out, const std::string &s, std::atomic_bool &b) {

  auto t = std::thread([s, &out, &b]() {

    // as long as we load
    int32_t dot = 0;
    while(!b) {

      out << '\r' << s;
      for(int32_t i = 0; i < dot; ++i) { out << '.';}
      dot = (dot + 1) % 4;
      usleep(300000);
    }

    out << '\n';
  });

  return std::move(t);
}

void load_binary_command(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

  // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Loading the file", b);

  // try to deserialize
  bbts::parsed_command_list_t cmd_list;
  bool success = cmd_list.deserialize(file_path);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!success) {
    out << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return;
  }

  // kick off a loading message
  b = false;
  t = loading_message(out, "Scheduling the loaded commands", b);

  // load the commands we just parsed
  auto [did_load, message] = node.load_commands(cmd_list);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to schedule the loaded commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}

void compile_commands(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

    // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Compiling commands", b);

  // compile the commands and load them
  auto [did_compile, message] = node.compile_commands(file_path);

  // finish the loading message
  b = true; t.join();

  if(!did_compile) {
    out << bbts::red << "Failed to compile the : \"" << message << "\"\n" << bbts::reset;
  } else {
    out << bbts::green << message << bbts::reset;
  }
}

bool load_shared_library(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

  // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Loading the library file", b);

  // try to open the file
  std::ifstream in(file_path, std::ifstream::ate | std::ifstream::binary);

  if(in.fail()) {
    // finish the loading message
    b = true; t.join();

    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return false;
  }

  auto file_len = (size_t) in.tellg();
  in.seekg (0, std::ifstream::beg);

  auto file_bytes = new char[file_len];
  in.readsome(file_bytes, file_len);

  // finish the loading message
  b = true; t.join();

  // kick off a registering message
  b = false;
  t = loading_message(out, "Registering the library", b);

  auto [did_load, message] = node.load_shared_library(file_bytes, file_len);
  delete[] file_bytes;

  // finish the registering message
  b = true; t.join();

  if(!did_load) {
    out << bbts::red << "Failed to register the library : \"" << message << "\"\n" << bbts::reset;
    return false;
  } else {
    out << bbts::green << message << bbts::reset;
    return true;
  }


}

void run_commands(std::ostream &out, bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Running the commands", b);

  // run all the commands
  auto [did_load, message] = node.run_commands();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to run commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}

void print(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Loading the file", b);

  // try to deserialize
  bbts::parsed_command_list_t cmd_list;
  bool success = cmd_list.deserialize(file_path);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!success) {
    out << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
  }

  // print out the commands
  cmd_list.print(out);
}

void clear(std::ostream &out, bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Clearing", b);

  // run all the commands
  auto [did_load, message] = node.clear();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to clear : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}

void set(std::ostream &out, bbts::node_t &node, const std::string &what, const std::string &value) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Setting", b);

  // parse the number of threads
  if(what == "no_threads") {

    // check the value
    if(value.empty()) {
      out << "You must provide a number of threads.\n";
    }

    // get the value
    char *p; auto val = strtoul(value.c_str(), &p, 10);
    auto [did_load, message] = node.set_num_threads(val);

    // finish the loading message
    b = true; t.join();

    // did we fail
    if(!did_load) {
      out << bbts::red << "Failed to set : \"" << message << "\"\n" << bbts::reset;
    }
    else {
      out << bbts::green << message << bbts::reset;
    }
  }
  else if(what == "max_mem") {

    // check the value
    if(value.empty()) {

      // finish the loading message
      b = true; t.join();
      out << "You must provide a number of threads.\n";

      return;
    }

    // seperate the unit from the value
    auto num = value; num.pop_back();
    char unit = value.back();

    // check the unit
    if(!(unit == 'K' || unit == 'M' || unit == 'G') || num.empty()) {

      // finish the loading message
      b = true; t.join();
      out << bbts::red  << "The value must be a positive number followed by [K|M|G].\n" << bbts::reset;

      return;
    }

    // get the value
    char *p; auto val = strtoull(num.c_str(), &p, 10);

    // apply the unit
    switch(unit) {
      case 'K' : val *= 1024; break;
      case 'M' : val *= (1024 * 1024); break;
      case 'G' : val *= (1024 * 1024 * 1024); break;
      default : break;
    }

    // set the value
    auto [did_load, message] = node.set_max_storage(val);

    // finish the loading message
    b = true; t.join();

    // did we fail
    if(!did_load) {
      out << bbts::red << "Failed to set : \"" << message << "\"\n" << bbts::reset;
    }
    else {
      out << bbts::green << message << bbts::reset;
    }
  }
  else {

    // finish the loading message
    b = true; t.join();

    out << bbts::red << "You can only set :\n" << bbts::reset;
    out << bbts::red << "no_threads - the number of threads\n" << bbts::reset;
    out << bbts::red << "max_mem - the maximum memory\n" << bbts::reset;
  }
}

void verbose(std::ostream &out, bbts::node_t &node, bool val) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Set verbose", b);

  // run all the commands
  auto [did_load, message] = node.set_verbose(val);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Set to fail verbose : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}

void shutdown(std::ostream &out, bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Shutting down", b);

  // run all the commands
  auto [did_load, message] = node.shutdown_cluster();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to shutdown : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}


// the prompt
void prompt(bbts::node_t &node) {

  std::cout << "\n";
  std::cout << "\t\t    \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \n";
  std::cout << "\t\t-----///----///----///----///----///----///----///----///-----\n";
  std::cout << "\t\t    /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  \n";
  std::cout << "\n";
  std::cout << "\t\t\tWelcome to " << bbts::green << "BarbaTOS" << bbts::reset << ", the tensor operating system\n";
  std::cout << "\t\t\t\tVersion : 0.1 - Lupus Rex\n";
  std::cout << "\t\t\t\tEmail : dj16@rice.edu\n";
  std::cout << '\n';

  auto rootMenu = std::make_unique<Menu>("cli");

  // set up load commands and load library
  auto loadSubMenu = std::make_unique<Menu>("load");

  loadSubMenu->Insert("commands",[&](std::ostream &out, const std::string &file) {

    load_binary_command(out, node, file);

  },"Load commands form a binary file. Usage : load commands <file>\n");

  loadSubMenu->Insert("library", [&](std::ostream &out, const std::string &file) {

    load_shared_library(out, node, file);

   },"Load a shared object file. Usage : load library <file>\n");

  rootMenu->Insert(std::move(loadSubMenu));

  // setup the info command
  rootMenu->Insert("info",[&](std::ostream &out, const std::string &what) {

    if(what == "cluster") {
      node.print_cluster_info(out);
    }
    else if(what == "storage") {

      auto [success, message] = node.print_storage_info();
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << message << '\n';
    }

  },"Returns information about the cluster. Usage : info [cluster, storage, tensor] \n ");

  rootMenu->Insert("info",[&](std::ostream &out, const std::string &what, int32_t id) {

    if(what == "tensor") {
      auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(id));
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << message << '\n';
    }

  },"Returns information about the cluster. Usage : info [cluster, storage, tensor] [tid] \n ");

  rootMenu->Insert("run",[&](std::ostream &out) {

    run_commands(out, node);

  },"Run scheduled commands.\n");

  rootMenu->Insert("verbose",[&](std::ostream &out, bool val) {

    verbose(out, node, val);

  },"Enables or disables debug messages. verbose [true|false]\n");

  rootMenu->Insert("print",[&](std::ostream &out, const std::string &file) {

    print(out, node, file);

  },"Prints command stored in a file. Usage : print <file>\n");


  auto compileSubMenu = std::make_unique<Menu>("compile");

  compileSubMenu->Insert("raw",[&](std::ostream &out, const std::string &file) {

    compile_commands(out, node, file);

  },"Compiles a command raw .sbbts file. Usage : compile raw <file>\n");

  rootMenu->Insert(std::move(compileSubMenu));

  rootMenu->Insert("clear",[&](std::ostream &out) {

    clear(out, node);

  },"Clears the tensor operating system.\n");

  rootMenu->Insert("a", [&](std::ostream &out) {

    compile_commands(out, node, "cmds/setup.barb");
    run_commands(out, node);

  }, "compile and run cmds/setup.barb\n");

  rootMenu->Insert("b", [&](std::ostream &out) {

    compile_commands(out, node, "cmds/run.barb");
    run_commands(out, node);

  }, "compile and run cmds/run.barb\n");

  rootMenu->Insert("set",[&](std::ostream &out, const std::string &what, const std::string &val) {

    set(out, node, what, val);

  },"Sets a value in the system. Usage : set <no_threads, max_mem> <value>.\n");


  // init the command line interface
  Cli cli(std::move(rootMenu));

  // global exit action
  cli.ExitAction([&](auto &out) { shutdown(out, node); });

  // start the cli session
  CliFileSession input(cli);
  input.Start();
}

int main(int argc, char **argv) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv, .num_threads = 8});

  // create the node
  bbts::node_t node(config);

  // init the node
  node.init();

  // sync everything
  node.sync();

  // kick off the prompt
  std::thread t;
  if (node.get_rank() == 0) {
    t = std::thread([&]() {
      load_shared_library(
          std::cout,
          node,
          "bbts/libraries/libbarbcu.so");

      for(int i = 0; i != 4; ++i) {
        compile_commands(std::cout, node, "cmds/setup.barb");
        run_commands(std::cout, node);
        compile_commands(std::cout, node, "cmds/run.barb");
        run_commands(std::cout, node);
        clear(std::cout, node);
      }

      shutdown(std::cout, node);
    });
  }

  // the node
  node.run();

  // wait for the prompt to finish
  if (node.get_rank() == 0) { t.join();}

  return 0;
}

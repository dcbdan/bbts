#include <iostream>
#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/server/node.h"
#include "../src/commands/parsed_command.h"
#include "../src/utils/terminal_color.h"

#include "../third_party/cli/include/cli/cli.h"
#include "../third_party/cli/include/cli/clifilesession.h"

using namespace cli;

std::thread loading_message(const std::string &s, std::atomic_bool &b) {

  auto t = std::thread([s, &b]() {

    // as long as we load
    int32_t dot = 0;
    while(!b) {

      std::cout << '\r' << s;
      for(int32_t i = 0; i < dot; ++i) { std::cout << '.';}
      dot = (dot + 1) % 4;
      usleep(300000);
    }

    std::cout << '\n';
  });

  return std::move(t);
}

void load_binary_command(bbts::node_t &node, const std::string &file_path) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message("Loading the file", b);

  // try to deserialize
  bbts::parsed_command_list_t cmd_list;
  bool success = cmd_list.deserialize(file_path);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!success) {
    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
  }

  // kick of a loading message
  b = false;
  t = loading_message("Scheduling the loaded commands", b);

  // load the commands we just parsed
  auto [did_compile, error] = node.load_commands(cmd_list);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_compile) {
    std::cout << bbts::red << "Failed to schedule the loaded commands : \"" << error << "\"\n" << bbts::reset;
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

  // setup the info command
  rootMenu->Insert("info",
                   [&](std::ostream &out) { node.print_cluster_info(out); },
                   "Returns information about the cluster\n");

  rootMenu->Insert("load",[&](std::ostream &out, const std::string &file) {

    load_binary_command(node, file);

  },"Load commands form a binary file. Usage : load <file>\n");

  // init the command line interface
  Cli cli(std::move(rootMenu));

  // global exit action
  cli.ExitAction([](auto &out) { out << "Goodbye...\n"; });

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
  if (node.get_rank() == 0) {
    std::thread t = std::thread([&]() { prompt(node); });
    t.detach();
  }

  // the node
  node.run();

  return 0;
}

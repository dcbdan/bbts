#include "parse_args.h"

#ifdef ENABLE_IB
#include <fstream>
#include <iostream>
#endif

namespace bbts {

#ifdef ENABLE_IB

std::tuple<int, connection_info_t> parse_connection_args(int argc, char** argv) {
  std::string usage = "usage: " + std::string(argv[0]) + " <rank> <device name> <hosts file>";
  if(argc < 4) {
    throw std::runtime_error(usage);
  }

  int rank = std::stoi(argv[1]);

  std::vector<std::string> ips;
  {
    std::ifstream s = std::ifstream(argv[3]);
    if(!s.is_open()) {
      usage = "Coud not open '" + std::string(argv[3]) + "'\n" + usage;
      throw std::runtime_error(usage);
    }

    std::string l;
    while(std::getline(s, l)) {
      if(l.size() < 7) {
        usage = "Invalid hosts file\n" + usage;
        throw std::runtime_error(usage);
      }
      ips.push_back(l);
    }
    for(auto i: ips){
      std::cout << "IP: " << i << std::endl;
    }
  }

  return {3, {std::string(argv[2]), rank, ips}};
}

#else

std::tuple<int, connection_info_t> parse_connection_args(int argc, char** argv) {
  return {0, {}};
}

#endif

}

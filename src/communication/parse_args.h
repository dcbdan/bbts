#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace bbts {
#ifdef ENABLE_IB
using connection_info_t = std::tuple<std::string, int32_t, std::vector<std::string>>;
std::tuple<int, connection_info_t> parse_connection_args(int argc, char** argv);
#else
using connection_info_t = std::tuple<>;
std::tuple<int, connection_info_t> parse_connection_args(int argc, char** argv);
#endif
}

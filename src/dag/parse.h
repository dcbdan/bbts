#pragma once

#include "dag.h"

namespace bbts { namespace dag {

void parse_dag_into(std::string filename, std::vector<node_t>& ret);

std::vector<node_t> parse_dag(std::string filename);

void print_dag(std::vector<node_t> const& dag);

}}

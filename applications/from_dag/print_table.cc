#include "print_table.h"

table_t::next_line_t table_t::endl{};

table_t::blank_t table_t::blank{};

table_t& operator<<(table_t& t, table_t::next_line_t const&)
{
  t.table.push_back(std::vector<std::string>());
  t.table.back().reserve(80);
  return t;
}

table_t& operator<<(table_t& t, table_t::blank_t const&)
{
  t.table.back().push_back("");
  return t;
}

std::ostream& operator<<(std::ostream& os, table_t const& t) {
  t.print(os);
  return os;
}

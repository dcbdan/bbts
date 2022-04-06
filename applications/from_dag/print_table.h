#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

class table_t {
  std::vector<std::vector<std::string>> table;

  struct next_line_t {};
  struct blank_t {};

  friend table_t& operator<<(table_t& t, next_line_t const&);
  friend table_t& operator<<(table_t& t, blank_t const&);

  int const tab_size;

public:
  static next_line_t endl ;
  static blank_t     blank;

  table_t(): table_t(2) {}

  table_t(int tab_size): table(1), tab_size(tab_size) {}

  template <typename T>
  void insert(T const& obj)
  {
    std::stringstream ss;
    ss << obj;

    table.back().push_back(ss.str());
  }

  void insert(std::string const& str) {
    table.back().push_back(str);
  }

  void print(std::ostream& os) const {
    std::vector<int> col_sizes;
    col_sizes.reserve(80);
    for(std::vector<std::string> const& line: table) {
      if(col_sizes.size() < line.size()) {
        col_sizes.resize(line.size(), 0);
      }
      for(int i = 0; i != line.size(); ++i) {
        if(line[i].size() > col_sizes[i]) {
          col_sizes[i] = line[i].size();
        }
      }
    }

    for(int which_line = 0; which_line != table.size(); ++which_line) {
      std::vector<std::string> const& line = table[which_line];

      for(int i = 0; i != line.size(); ++i) {
        std::string const& word = line[i];
        int const& col_size = col_sizes[i];

        int pad_size = tab_size + col_size - word.size();

        os << word << std::string(pad_size, ' ');
      }

      if(which_line + 1 != table.size()) {
        os << std::endl;
      }
    }
  }
};

template <typename T>
table_t& operator<<(table_t& t, T const& obj)
{
  t.insert(obj);
  return t;
}

table_t& operator<<(table_t& t, table_t::next_line_t const&);

table_t& operator<<(table_t& t, table_t::blank_t const&);

std::ostream& operator<<(std::ostream& os, table_t const& t);

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& xs)
{
  if(xs.size() == 0) {
    return os;
  }

  os << "[";
  for(int i = 0; i != xs.size() - 1; ++i) {
    os << xs[i] << ",";
  }
  os << xs.back() << "]";

  return os;
}




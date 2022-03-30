#pragma once

#include <vector>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <iostream>

#include "../commands/command_utils.h"

namespace bbts { namespace dag {

struct param_t {
  enum which_t {
    I, F, B
  };
 which_t  which;

  union val_t {
    int i;
    float f;
    bool b;
  };
  val_t val;
};

}}

std::ostream& operator<<(std::ostream& os, bbts::dag::param_t p);
std::istream& operator>>(std::istream& is, bbts::dag::param_t& p);

namespace bbts { namespace dag {

struct node_t {
  using nid_t  = int;
  using dim_t  = int;
  using rank_t = int;

  enum node_type {
    input,
    reblock,
    join,
    agg
  };

  node_type type;

  nid_t id; // this node itself

  // The dimensions for agg and reblock nodes can be deduced
  // from the join nodes. Nevertheless, store them anyway.
  std::vector<dim_t> dims;

  std::vector<nid_t> downs;
  std::vector<nid_t> ups;

  std::vector<param_t> params;

  // only valid if type == join
  std::vector<std::vector<rank_t>> ordering;
  std::vector<rank_t> aggs;


private:
  friend std::ostream& operator<<(std::ostream& os, node_t const& self) {
    return self.print(os);
  }

  std::ostream& print(std::ostream& os) const;
};

template <typename T>
void print_list(std::ostream& os, std::vector<T> xs) {
  if(xs.size() > 0) {
    os << xs[0];
  }
  for(int i = 1; i < xs.size(); ++i) {
    os << "," << xs[i];
  }
}

}}

#pragma once

#include <vector>
#include <tuple>
#include <cassert>

#include "expand.h" // this is for expand_dim_t

namespace utils { namespace expand {

using std::vector;
using std::tuple;

struct expand_indexer_t {
  expand_indexer_t(
    vector<int> const& num_block_inn,
    vector<int> const& num_block_out):
      num_block_inn(num_block_inn),
      num_block_out(num_block_out)
  {
    assert(num_block_inn.size() == num_block_out.size());
  }

  // get the cartesian product of all closed ranges
  static vector<vector<int>> cartesian(vector<tuple<int,int>> const& rs);
  // the same thing except the interval are given by [0,szs[0]-1], ..., [0,szs[n]-1]
  static vector<vector<int>> cartesian(vector<int> const& szs);

  // cartesian(rs)[which] == uncartesian(rs, which)
  static vector<int> uncartesian(vector<tuple<int,int>> const& rs,  int which);
  static vector<int> uncartesian(vector<int>            const& szs, int which);

  // get a list of inputs that are required from this particular output
  vector<tuple<int,int>>
  get_inputs(vector<int> which_out) const;

  // uncartesian(get_inputs(which_out), index_which_inn)
  //   == cartesian(get_inputs(which_out))[index_which_inn]
  // The benefit of using get_which_input is that it doesn't call cartesian
  vector<int> get_which_input(vector<int> const& which_out, int index_which_inn) const;

  vector<expand_dim_t> get_expand_dim(
    vector<int> const& dim,
    vector<int> const& which_inn,
    vector<int> const& which_out) const;
  vector<expand_dim_t> get_expand_dim(
    vector<int> const& dim,
    int index_which_inn,
    vector<int> const& which_out) const;

  // given a particular dim, get which blocks of that dim are required.
  // For
  //   auto [s,e] = dim_which(which_dim, which_out)
  // the input indices s,...,e are required by the output
  tuple<int,int> dim_which(int which_dim, int which_out) const;

  int get_num_dims() const { return num_block_inn.size(); }

  vector<int> const num_block_inn;
  vector<int> const num_block_out;
};

///////////////////
//#include <iostream>
//#include <string>
//#include "expand_indexer.h"
//
//using std::string;
//using namespace utils::expand;
//
//void print_interval(string s, int u, int v, string t) {
//  std::cout << s << "[" << u << ", " << v << ")" << t;
//}
//void print_interval(int u, int v) {
//  print_interval("", u, v, "");
//}
//void print_vec(vector<int> xs) {
//  for(int x: xs) {
//    std::cout << x << " ";
//  }
//}
//
//void test_vec(int inn, int out) {
//  vector<int> num_inn = {inn};
//  vector<int> num_out = {out};
//  expand_indexer_t s(num_inn, num_out);
//
//  for(int i = 0; i != out; ++i) {
//    auto [x,y] = s.dim_which(0, i);
//    print_interval("out: ", i*inn, (i+1)*inn, "\n");
//
//    std::cout << "ins: ";
//    for(int j = x; j <= y; ++j) {
//      print_interval(j*out, (j+1)*out);
//    }
//    std::cout << std::endl;
//
//    std::cout << "which: ";
//    auto [u,v] = s.get_inputs({i})[0];
//    print_interval("which: ", u, v+1, "\n");
//  }
//}
//
//int main()
//{
//  test_vec(3,4);
//  std::cout << "-------------------------------------------------\n";
//  test_vec(4,3);
//  std::cout << "-------------------------------------------------\n";
//  {
//    expand_indexer_t indexer({4}, {3});
//    auto [x,y] = indexer.get_inputs({0})[0];
//    std::cout << x << ", " << y << std::endl;
//  }
//
//  //{
//  //  int ix = 10;
//  //  int iy = 11;
//  //  int iz = 12;
//
//  //  int nx = 2;
//  //  int ny = 3;
//  //  int nz = 4;
//
//  //  int nn = nx*ny*nz;
//
//  //  vector<tuple<int,int>> rs{{ix, ix + nx - 1}, {iy, iy + ny - 1}, {iz, iz + nz - 1}};
//
//  //  for(int i = 0; i != nn; ++i) {
//  //    std::cout << i << ": ";
//  //    print_vec(expand_indexer_t::cartesian(rs)[i]); std::cout << std::endl;
//  //  }
//  //  for(int i = 0; i != nn; ++i) {
//  //    std::cout << std::boolalpha <<
//  //      (expand_indexer_t::cartesian(rs)[i] == expand_indexer_t::uncartesian(rs, i)) <<
//  //      std::endl;
//  //  }
//  //}
//}

} // expand
} // utils

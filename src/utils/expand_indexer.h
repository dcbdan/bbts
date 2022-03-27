#pragma once

#include <vector>
#include <tuple>
#include <cassert>

#include "expand.h" // this is form expand_dim_t

namespace utils { namespace expand {

using std::vector;
using std::tuple;

struct expand_indexer_t {
  expand_indexer_t(
    vector<int> const& num_block_inn,
    vector<int> const& num_block_out):
      num_block_inn(num_block_inn),
      num_block_out(num_block_out)
  {}

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

  vector<int> const& num_block_inn;
  vector<int> const& num_block_out;
};

vector<tuple<int,int>>
expand_indexer_t::get_inputs(vector<int> which_out) const
{
  int num_dims = get_num_dims();

  vector<tuple<int,int>> ret;
  ret.reserve(num_dims);

  for(int r = 0; r != num_dims; ++r) {
    ret.push_back(dim_which(r, which_out[r]));
  }

  return ret;
}

vector<vector<int>>
expand_indexer_t::cartesian(vector<tuple<int,int>> const& rs)
{
  int n = 1;

  vector<int> idx;
  idx.reserve(rs.size());

  for(auto const& [s,e]: rs) {
    idx.push_back(s);
    n *= (e - s + 1);
  }

  // idx is being incremented along a column major ordering
  auto increment = [&]() {
    for(int i = 0; i != rs.size(); ++i) {
      if(idx[i] == std::get<1>(rs[i])) {
        idx[i] = std::get<0>(rs[i]);
      } else {
        idx[i] += 1;
        break;
      }
    }
  };

  vector<vector<int>> ret(n, vector<int>(rs.size()));
  for(int i = 0; i != n; ++i) {
    ret[i] = idx;
    increment();
  }

  return ret;
}

vector<vector<int>>
expand_indexer_t::cartesian(vector<int> const& szs)
{
  vector<tuple<int,int>> rs(szs.size());
  std::transform(
    szs.begin(),
    szs.end(),
    rs.begin(),
    [](int const& sz) { return tuple<int,int>{0,sz-1}; });
  return expand_indexer_t::cartesian(rs);
}

vector<int>
expand_indexer_t::uncartesian(
  vector<tuple<int,int>> const& rs,
  int which)
{
  vector<int> cums;
  cums.reserve(rs.size());
  cums.push_back(1);
  for(int i = 0; i < rs.size()-1; ++i) {
    auto const& [s,e] = rs[i];
    cums.push_back(cums.back() * (e - s + 1));
  }

  vector<int> ret(rs.size());
  for(int i = rs.size() - 1; i >= 0; --i) {
    ret[i] = std::get<0>(rs[i]) + (which / cums[i]);
    which =                        which % cums[i];
  }
  return ret;
}

vector<int>
expand_indexer_t::uncartesian(
  vector<int> const& szs,
  int which)
{
  vector<tuple<int,int>> rs(szs.size());
  std::transform(
    szs.begin(),
    szs.end(),
    rs.begin(),
    [](int const& sz) { return tuple<int,int>{0,sz-1}; });
  return expand_indexer_t::uncartesian(rs, which);
}

vector<int>
expand_indexer_t::get_which_input(
  vector<int> const& which_out,
  int index_which_inn) const
{
  vector<tuple<int,int>> rngs = get_inputs(which_out);
  return uncartesian(rngs, index_which_inn);
}

vector<expand_dim_t>
expand_indexer_t::get_expand_dim(
  vector<int> const& dim,
  vector<int> const& which_inn,
  vector<int> const& which_out) const
{
  vector<expand_dim_t> ret;
  ret.reserve(num_block_inn.size());

  for(int i = 0; i != num_block_inn.size(); ++i) {
    int const& d      = dim[i];
    int const& w_inn  = which_inn[i];
    int const& w_out  = which_out[i];
    int const& nb_inn = num_block_inn[i];
    int const& nb_out = num_block_out[i];

    int kd_inn = d / nb_inn;
    int kd_out = d / nb_out;
    // the input, output intervals are
    //   [w_inn*kd_inn, (w_inn+1)*kd_inn) and
    //   [w_out*kd_out, (w_out+1)*kd_out)
    // Take the intersection of the two intervals to get
    //   [s,e)
    int s = std::max(w_inn*kd_inn,
                     w_out*kd_out);
    int e = std::min((w_inn+1)*kd_inn,
                     (w_out+1)*kd_out);
    ret.push_back(expand_dim_t{
      .interval  = e - s,
      .start_inn = s - (w_inn*kd_inn),
      .full_inn  = kd_inn,
      .start_out = s - (w_out*kd_inn),
      .full_out  = kd_out
    });
  }

  return ret;
}

vector<expand_dim_t>
expand_indexer_t::get_expand_dim(
  vector<int> const& dim,
  int index_which_inn,
  vector<int> const& which_out) const
{
  return get_expand_dim(
    dim,
    get_which_input(which_out, index_which_inn),
    which_out);
}

// given a particular dim, get which input blocks of that dim are required
tuple<int, int>
expand_indexer_t::dim_which(int which_dim, int which_out) const
{
  int const& inn = num_block_inn[which_dim];
  int const& out = num_block_out[which_dim];

  tuple<int,int> ret;
  auto& [ret_low,ret_upp] = ret;

  // the inn blocking is [0,out)[out,2*out),...,[(inn-1)*out,inn*out)
  // the out blocking is [0,inn)[inn,2*inn),...,[(out-1)*inn,out*inn)

  // Determine which inputs intersect I := [which_out*inn, (which_out+1)*inn).

  // The first input is
  //   min i such that [i*out, (i+1)*out) `intersect` I /= empty set.
  // That is equal to
  //   min i such that (i+1)*out > which_out*inn
  //                   (i+1)*y   > j         *x
  //   (i+1)y > jx
  //   iy + y > jx
  //   iy > jx - y
  //   i  > jx / y - 1
  //
  {
    int& i = ret_low;
    i = which_out * inn / out - 1;
    if((i+1)*out <= which_out*inn) {
      i++;
    }
    //assert((i+1)*out > which_out*inn && i*out <= which_out*inn);
  }

  // The last output is
  //   max i such that [i*out, (i-1)*out) `intersect` I /= empty set.
  // That is equal to
  //   max i such that i*out < (which_out+1)*inn
  //                   i*y   < (j        +1)*x
  //   iy < (j+1)*x
  //   iy < jx+x
  //   i  < (jx+x)/y
  {
    int& i = ret_upp;
    i = (which_out*inn + inn) / out;
    if(i*out >= (which_out+1)*inn) {
      i--;
    }
    //assert(i*out < (which_out+1)*inn && (i+1)*out >= (which_out+1)*inn);
  }

  return ret;
}

///////////////////
//#include <iostream>
//#include <string>
//
//using std::string;
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
//int main()
//{
//  {
//    int inn = 3;
//    int out = 4;
//    vector<int> num_inn = {inn};
//    vector<int> num_out = {out};
//    expand_indexer_t s(num_inn, num_out);
//
//    for(int i = 0; i != out; ++i) {
//      auto [x,y] = s.dim_which(0, i);
//      print_interval("out: ", i*inn, (i+1)*inn, "\n");
//
//      std::cout << "ins: ";
//      for(int j = x; j <= y; ++j) {
//        print_interval(j*out, (j+1)*out);
//      }
//      std::cout << std::endl;
//    }
//  }
//
//  {
//    int ix = 10;
//    int iy = 11;
//    int iz = 12;
//
//    int nx = 2;
//    int ny = 3;
//    int nz = 4;
//
//    int nn = nx*ny*nz;
//
//    vector<tuple<int,int>> rs{{ix, ix + nx - 1}, {iy, iy + ny - 1}, {iz, iz + nz - 1}};
//
//    for(int i = 0; i != nn; ++i) {
//      std::cout << i << ": ";
//      print_vec(expand_indexer_t::cartesian(rs)[i]); std::cout << std::endl;
//    }
//    for(int i = 0; i != nn; ++i) {
//      std::cout << std::boolalpha <<
//        (expand_indexer_t::cartesian(rs)[i] == expand_indexer_t::uncartesian(rs, i)) <<
//        std::endl;
//    }
//  }
//}

} // expand
} // utils

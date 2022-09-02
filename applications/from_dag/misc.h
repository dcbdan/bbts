#pragma once

#include <vector>
#include <cstdint>

namespace bbts { namespace dag {

// TODO(cleanup): this is a source of bugs:
//   myint = product(vector<int>{...}) is a silent killer..
// Use product_to_int, product_to_uint64_t and so on...
template <typename T, typename U=uint64_t>
U product(std::vector<T> const& xs) {
  U ret = 1;
  for(auto x: xs) {
    ret *= x;
  }
  return ret;
}

struct indexer_t {
  indexer_t(std::vector<int> max):
    max(max), idx(max.size())
  {}

  bool increment(){
    bool could_increment = false;
    for(int i = 0; i != max.size(); ++i) {
      if(idx[i] + 1 == max[i]) {
        idx[i] = 0;
      } else {
        idx[i] += 1;
        could_increment = true;
        break;
      }
    }
    return could_increment;
  }

  std::vector<int> const max;
  std::vector<int>       idx;
};

std::vector<std::vector<int>> cartesian(
  std::vector<std::vector<int> > const& vs);

std::vector<int> expand1(std::vector<int> const& xs);
std::vector<int> expand0(std::vector<int> const& xs);
std::vector<int> squeeze(std::vector<int> const& xs);

}}

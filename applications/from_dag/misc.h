#pragma once

#include <vector>

namespace bbts { namespace dag {

template <typename T>
T product(std::vector<T> const& xs) {
  T ret = 1;
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

}}

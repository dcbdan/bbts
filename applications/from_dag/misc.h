#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

#define PRINT_VEC(x, vec)   \
  std::cout << x << " ";    \
  for(auto const& v: vec) { \
    std::cout << v << " ";  \
  } std::cout << std::endl

namespace bbts { namespace dag {

// TODO(cleanup): this is a source of bugs:
//   myint = product(vector<int>{...}) is a silent killer..
// Use product_to_int, product_to_uint64_t and so on...
template <typename T, typename U=uint64_t>
U product(std::vector<T> const& xs) {
  U ret = 1;
  for(auto const& x: xs) {
    ret *= x;
  }
  return ret;
}

template <typename T>
int product_to_int(std::vector<T> const& xs) {
  uint64_t ret = 1;
  for(auto const& x: xs) {
    ret *= x;
  }
  uint64_t m = static_cast<uint64_t>(std::numeric_limits<int>::max());
  if(ret > m) {
    throw std::runtime_error("product_to_int: the inputs are too big");
  }
  return static_cast<int>(ret);
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

void dcb_assert(bool b);

}}

#include "misc.h"

namespace bbts { namespace dag {

std::vector<std::vector<int>> cartesian(
  std::vector<std::vector<int> > const& vs)
{
  // assuming all vs have a size greater than zero

  std::vector<int> max;
  for(auto const& v: vs) {
    max.push_back(v.size());
  }
  indexer_t indexer(max);

  std::vector<std::vector<int>> ret;
  do {
    std::vector<int> next_val;
    next_val.reserve(vs.size());
    for(int which_v = 0; which_v != vs.size(); ++ which_v) {
      next_val.push_back(vs[which_v][indexer.idx[which_v]]);
    }
    ret.push_back(next_val);
  } while(indexer.increment());

  return ret;
}

std::vector<int> expand1(std::vector<int> const& xs)
{
  std::vector<int> ret(xs.size() + 1);
  ret[0] = 1;
  std::copy(xs.begin(), xs.end(), ret.begin() + 1);
  return ret;
}

std::vector<int> expand0(std::vector<int> const& xs)
{
  std::vector<int> ret(xs.size() + 1);
  ret[0] = 0;
  std::copy(xs.begin(), xs.end(), ret.begin() + 1);
  return ret;
}

std::vector<int> squeeze(std::vector<int> const& xs)
{
  return std::vector<int>(xs.begin() + 1, xs.end());
}

void dcb_assert(bool b) {
  if(!b) {
    throw std::runtime_error("Failed dcb_assert");
  }
}

}}

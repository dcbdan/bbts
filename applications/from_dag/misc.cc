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


}}

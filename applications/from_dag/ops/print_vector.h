#pragma once

#include <iostream>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& xs) {
  if(xs.size() == 0) {
    os << "[]";
    return os;
  }

  os << "[" << xs[0];
  for(int i = 1; i < xs.size(); ++i) {
    os << "," << xs[i];
  }
  os << "]";

  return os;
}



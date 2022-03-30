#pragma once

#include <functional>

template <typename T>
struct cache_holder_t {
  cache_holder_t(std::function<T()> on_access):
    on_access(on_access),
    set(false)
  {}

  T const& operator()() const {
    if(!set) {
      val = on_access();
      set = true;
    }
    return val;
  }

private:
  std::function<T()> on_access;
  mutable bool set;
  mutable T val;
};


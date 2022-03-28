#pragma once

#include <tuple>
#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>

namespace utils { namespace expand {

using std::tuple;
using std::vector;
using std::function;

// [start_inn, start_inn + interval) corresponds to [start_out, start_out + interval)
// The dimension of the inn and out are [0,full_inn) and [0,full_out)
struct expand_dim_t {
  int interval;
  int start_inn, full_inn;
  int start_out, full_out;
};

// This correspondonds to [start, start + interval) which is a subset of [0,full).
struct part_dim_t {
  int start;
  int interval;
  int full;

  bool is_full() const {
    return start == 0 && interval == full;
  }
  bool is_not_full() const {
    return start != 0 || interval != full;
  }
};

}}

std::ostream& operator<<(std::ostream& out, utils::expand::part_dim_t const& p);
std::ostream& operator<<(std::ostream& out, utils::expand::expand_dim_t const& e);

namespace utils { namespace expand {

template <typename Indexer>
struct expand_t {
  expand_t(vector<expand_dim_t> const& expand_dims):
    expand_dims(expand_dims)
  {}

  static void for_each_offset(
    vector<part_dim_t> const& ps,
    function<void(int,int)> f)
  {
    Indexer idx(ps);
    int num_copy = idx.leading().interval;
    do {
      int offset = idx();
      f(offset, num_copy);
    } while (idx.increment());
  }

  static void small_to_big(
    float* data_inn,
    float* data_out,
    vector<part_dim_t> const& part_out)
  {
    for_each_offset(part_out,
      [&](int offset, int num_copy) {
        std::copy(data_inn, data_inn + num_copy, data_out + offset);
        data_inn += num_copy;
      });
  }

  static void big_to_small(
    float* data_inn,
    vector<part_dim_t> const& part_inn,
    float* data_out)
  {
    for_each_offset(part_inn,
      [&](int offset, int num_copy) {
        std::copy(data_inn + offset, data_inn + offset + num_copy, data_out);
        data_out += num_copy;
      });
  }

  static int parts_size(vector<part_dim_t> const& parts) {
    int ret = 1;
    for(auto const& p: parts) {
      ret *= p.interval;
    }
    return ret;
  }

  bool is_compact_inn() const {
    for(auto const& e: expand_dims) {
      if(e.start_inn != 0 || e.interval != e.full_inn) {
        return false;
      }
    }
    return true;
  }

  bool is_compact_out() const {
    for(auto const& e: expand_dims) {
      if(e.start_out != 0 || e.interval != e.full_out) {
        return false;
      }
    }
    return true;
  }

  vector<part_dim_t> get_parts_out() const {
    vector<part_dim_t> parts_out;
    parts_out.reserve(expand_dims.size());

    for(auto const& e: expand_dims) {
      parts_out.push_back({
        .start    = e.start_out,
        .interval = e.interval,
        .full     = e.full_out});
    }

    return parts_out;
  }

  vector<part_dim_t> get_parts_inn() const {
    vector<part_dim_t> parts_inn;
    parts_inn.reserve(expand_dims.size());

    for(auto const& e: expand_dims) {
      parts_inn.push_back({
        .start    = e.start_inn,
        .interval = e.interval,
        .full     = e.full_inn});
    }

    return parts_inn;
  }

  vector<int> compact_inn_shape() const {
    vector<int> shape;
    shape.reserve(expand_dims.size());
    for(part_dim_t const& p: get_parts_inn()) {
      shape.push_back(p.interval);
    }
    return shape;
  }

  vector<int> expand_out_shape() const {
    vector<int> shape;
    shape.reserve(expand_dims.size());
    for(expand_dim_t const& e: expand_dims) {
      shape.push_back(e.full_out);
    }
    return shape;
  }

  void compact(float* data_inn, float* data_small) const {
    vector<part_dim_t> parts_inn = get_parts_inn();
    Indexer::merge(parts_inn);
    big_to_small(data_inn, parts_inn, data_small);
  }

  void uncompact(float* data_inn, float* data_out) const {
    vector<part_dim_t> parts_out = get_parts_out();
    Indexer::merge(parts_out);
    small_to_big(data_inn, data_out, parts_out);
  }

  void expand(float* data_inn, float* data_out) const {
    vector<part_dim_t> parts_out = get_parts_out();
    Indexer::merge(parts_out);

    if(is_compact_inn()) {
      small_to_big(data_inn, data_out, parts_out);
      return;
    }

    vector<part_dim_t> parts_inn = get_parts_inn();
    Indexer::merge(parts_inn);

    int rect_sz = parts_size(parts_inn);
    float* tmp = new float[rect_sz];
    big_to_small(data_inn, parts_inn, tmp);
    small_to_big(tmp, data_out, parts_out);
    delete[] tmp;
  }

private:
  vector<expand_dim_t> const expand_dims;
};

struct column_major_indexer_t {
  column_major_indexer_t(std::vector<part_dim_t> const& part):
    part(part),
    idx(part.size(), 0)
  {}

  part_dim_t leading() { return part.front(); };

  int operator()() const {
    // COLUMN MAJOR.
    // The first indices "move the fastest"
    // i0 + s0*i1 + s0*s1*i2 + ... +  s0*...*s(n-1)*in
    // p0*i0 + p1*i1 + p2*i2 + ... + pn*in
    //
    // Here, s[idx] is given by sz[idx],
    // and   i[idx] is given by (start[i] + idx[i])
    int p = 1;
    int total = 0;
    for(int i = 0; i != idx.size(); ++i) {
      total += p*(part[i].start + idx[i]);
      p *= part[i].full;
    }
    return total;
  }
  // the idea is to iterate through
  // {0} x num[1] x ... x num[m]
  // The algorithm:
  // - increment from in to back
  // - when you can't increment, set to zero go back 1
  // 0 0
  // 1 0
  // 0 1
  // 1 1
  // 0 2
  // 1 2
  // -----
  // 0 0 0
  // 1 0 0
  // 0 1 0
  // 1 1 0
  // 0 0 1
  // .....
  // 1 1 1
  bool increment() {
    bool could_increment = false;
    // NOTE: keeping idx[0] to be zero all the time
    for(int i = 1; i < idx.size(); ++i) {
      if(idx[i] + 1 == part[i].interval) {
        idx[i] = 0;
      } else {
        idx[i] += 1;
        could_increment = true;
        break;
      }
    }
    return could_increment;
  }

  static void merge(vector<part_dim_t>& ps)
  {
    if(ps.size() <= 1 || ps[0].is_not_full()) {
      return;
    }

    part_dim_t new_p = ps[0];

    int num_full = 1;
    for(; num_full < ps.size(); ++num_full) {
      auto const& p = ps[num_full];

      if(p.is_not_full()) {
        new_p = part_dim_t{
          .start    = new_p.interval * p.start,
          .interval = new_p.interval * p.interval,
          .full     = new_p.full     * p.full
        };
        break;
      } else {
        int new_interval = new_p.interval * p.interval;
        new_p = part_dim_t {
          .start = 0,
          .interval = new_interval,
          .full     = new_interval
        };
      }
    }

    if(num_full == ps.size()) {
      // in this case, 1 big std copy would happen
      ps[0] = new_p;
      ps.resize(1);
    } else {
      // new_p contains info for num_full + 1 items
      // [0, 1, 2, 3, 4, 5]
      //  *  *  *  ^  +  +
      //  ^  +  +
      //  num_full = 3
      ps[0] = new_p;
      int num_out = ps.size() - num_full;
      std::copy(
        ps.begin() + 1 + num_full,
        ps.end(),
        ps.begin() + 1);
      ps.resize(num_out);
    }
  }

  vector<part_dim_t> const& part;
  vector<int> idx;
};

struct row_major_indexer_t {
  row_major_indexer_t(std::vector<part_dim_t> const& part):
    part(part),
    idx(part.size(), 0)
  {}

  part_dim_t leading() { return part.back(); };

  int operator()() const {
    // ROW MAJOR.
    // The last indices "move the fastest"
    // Here, s[idx] is given by sz[idx],
    // and   i[idx] is given by (start[i] + idx[i])
    int p = 1;
    int total = 0;
    for(int i = idx.size() - 1; i >= 0; --i) {
      total += p*(part[i].start + idx[i]);
      p *= part[i].full;
    }
    return total;
  }

  bool increment() {
    bool could_increment = false;
    // NOTE: keeping idx[idx.size()-1] to be zero all the time
    for(int i = idx.size()-2; i >= 0; --i) {
      if(idx[i] + 1 == part[i].interval) {
        idx[i] = 0;
      } else {
        idx[i] += 1;
        could_increment = true;
        break;
      }
    }
    return could_increment;
  }

  static void merge(vector<part_dim_t>& ps)
  {
    std::reverse(ps.begin(), ps.end());
    column_major_indexer_t::merge(ps);
    std::reverse(ps.begin(), ps.end());
  }

  vector<part_dim_t> const& part;
  vector<int> idx;
};

using column_major_expand_t = expand_t<column_major_indexer_t>;
using row_major_expand_t    = expand_t<row_major_indexer_t>;

///////////////////////////////////////////////////////////////////
//#include <numeric> // iota
//#include <iostream>
//#include <chrono>
//#include <cassert>
//
//using time_measurement_t = decltype(std::chrono::high_resolution_clock::now());
//
//template <typename T>
//T product(vector<T> const& ts) {
//  T ret = 1;
//  for(T const& t: ts) {
//    ret *= t;
//  }
//  return ret;
//}
//
//struct tensor_t {
//  tensor_t(std::vector<int> dims):
//    dims(dims)
//  {
//    int sz = product(dims);
//    data = new float[sz];
//    std::fill(data, data + sz, 0.0);
//  }
//
//  int size() const {
//    return product(dims);
//  }
//
//  ~tensor_t() {
//    delete[] data;
//  }
//
//  vector<int> dims;
//  float* data;
//};
//
//double time(function<void()> f) {
//  time_measurement_t start = std::chrono::high_resolution_clock::now();
//  f();
//  time_measurement_t end = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double, std::milli> dur = end-start;
//  return dur.count();
//}
//
//int main() {
//  {
//    tensor_t big_c({10, 10});
//    tensor_t big_r({10, 10});
//    {
//      float v = 0.0;
//
//      for(int j = 0; j != 10; ++j) {
//        for(int i = 0; i != 10; ++i) {
//          big_c.data[i + 10*j] = v;
//          big_r.data[10*i + j] = v;
//          v++;
//        }
//      }
//    }
//
//    {
//      tensor_t small_c({5, 5});
//      tensor_t small_r({5, 5});
//
//      column_major_expand_t::big_to_small(
//        big_c.data,
//        {{5, 5, 10}, {5, 5, 10}},
//        small_c.data);
//      row_major_expand_t::big_to_small(
//        big_r.data,
//        {{5, 5, 10}, {5, 5, 10}},
//        small_r.data);
//
//      for(int i = 0; i != 5; ++i) {
//        for(int j = 0; j != 5; ++j) {
//          assert(small_c.data[i + 5*j] == small_r.data[5*i + j]);
//          //std::cout << small_c.data[i + 5*j] << " " << small_r.data[5*i + j] << std::endl;
//        }
//      }
//    }
//
//    {
//      tensor_t small_c({2, 2});
//
//      column_major_expand_t::big_to_small(
//        big_c.data,
//        {{0, 2, 10}, {8, 2, 10}},
//        small_c.data);
//
//      // should be 80,81,90,91
//      assert(small_c.data[0] == 80.0);
//      assert(small_c.data[1] == 81.0);
//      assert(small_c.data[2] == 90.0);
//      assert(small_c.data[3] == 91.0);
//      //std::cout << 0 << " " << 8 << " " << small_c.data[0] << std::endl;
//      //std::cout << 1 << " " << 8 << " " << small_c.data[1] << std::endl;
//      //std::cout << 0 << " " << 9 << " " << small_c.data[2] << std::endl;
//      //std::cout << 1 << " " << 9 << " " << small_c.data[3] << std::endl;
//    }
//  }
//
//  {
//    tensor_t small({5, 5});
//    std::iota(small.data, small.data + small.size(), 100.0);
//
//    tensor_t big({10, 10});
//
//    column_major_expand_t::small_to_big(
//      small.data,
//      big.data,
//      {{5,5,10}, {5,5,10}});
//
//    int v = 100.0;
//
//    for(int j = 0; j != 10; ++j) {
//      for(int i = 0; i != 10; ++i) {
//        if(j < 5 || i < 5) {
//          assert(big.data[i + 10*j] == 0.0);
//        } else {
//          assert(big.data[i + 10*j] = v++);
//        }
//        //std::cout << i << " " << j << " " << big.data[i + 10*j] << std::endl;
//      }
//    }
//  }
//
//  {
//    auto show_part = [](part_dim_t const& p) {
//      std::cout
//        << "[0," << p.full << ") | ["
//        << p.start << "," << p.start+p.interval << ")" << std::endl;
//    };
//    auto parts_eq = [](vector<part_dim_t> const& lhs, vector<part_dim_t> const& rhs) {
//      if(lhs.size() != rhs.size()){
//        return false;
//      }
//      for(int i = 0; i != lhs.size(); ++i) {
//        if(lhs[i].start    == rhs[i].start &&
//           lhs[i].interval == rhs[i].interval &&
//           lhs[i].full     == rhs[i].full)
//        {
//
//        } else {
//          return false;
//        }
//      }
//      return true;
//    };
//
//    vector<part_dim_t> parts{{0,5,5}, {8, 2, 10}};
//    //for(auto const& p: parts) {
//    //  show_part(p);
//    //}
//    vector<part_dim_t> parts_c = parts;
//    column_major_indexer_t::merge(parts_c);
//    assert(parts_eq(parts_c, {{40, 10, 50}}));
//    //std::cout << "column major merge " << std::endl;
//    //for(auto const& p: parts_c) {
//    //  show_part(p);
//    //}
//    vector<part_dim_t> parts_r = parts;
//    row_major_indexer_t::merge(parts_r);
//    assert(parts_eq(parts_r, parts));
//    //std::cout << "row major merge " << std::endl;
//    //for(auto const& p: parts_r) {
//    //  show_part(p);
//    //}
//  }
//
//  // compute the ratio of time it takes to copy small vs one unit of big
//  {
//    tensor_t big({10240, 10240});
//    std::iota(big.data, big.data + big.size(), 0.0);
//
//    tensor_t big_({10240, 10240});
//
//    tensor_t small({10240, 128});
//
//    std::cout << "------------------------------\n";
//
//    double copy_big = time([&](){
//      std::copy(big.data, big.data + big.size(), big_.data);
//    });
//
//    double copy_small_0 = time([&]() {
//      column_major_expand_t::big_to_small(
//        big.data, {{0, 10240, 10240}, {1000, 128, 10240}}, small.data);
//    });
//
//    double copy_small_1 = time([&](){
//      column_major_expand_t::big_to_small(
//        big.data, {{1000, 128, 10240}, {0, 10240, 10240}}, small.data);
//    });
//
//    double copy_small_2 = time([&]() {
//      column_major_expand_t::small_to_big(
//        small.data, big.data, {{0, 10240, 10240}, {1000, 128, 10240}});
//    });
//
//    double copy_small_3 = time([&](){
//      column_major_expand_t::small_to_big(
//        small.data, big.data, {{1000, 128, 10240}, {0, 10240, 10240}});
//    });
//
//    std::cout << copy_big << "ms" << std::endl;
//    std::cout << "ratios: " <<
//      (copy_small_0 / copy_big) << ", " <<
//      (copy_small_1 / copy_big) << ", " <<
//      (copy_small_2 / copy_big) << ", " <<
//      (copy_small_3 / copy_big) << std::endl;
//  }
//}

} // expand
} // utils

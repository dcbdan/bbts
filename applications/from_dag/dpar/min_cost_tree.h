#include <functional>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace tree {

using std::function;
using std::tuple;
using std::vector;
using std::unordered_map;

template <typename T>
struct tree_t {
  T& operator[](int id){
    auto& [ret, _0] = info[id];
    return ret;
  }

  T const& operator[](int id) const {
    auto const& [ret, _0] = info.at(id);
    return ret;
  }

  vector<int> const& children(int id) const { return info.at(id).children; }

  void insert_root(int id, T const& t) {
    info.insert({ id, info_t{ t, vector<int>{} }});
    _top_node = id;
  }

  void insert_child(int parent, int id, T const& t) {
    info.insert({ id, info_t{ t, vector<int>{} }});
    info[parent].children.push_back(id);
  }

  void insert_child(int parent, tree_t<T> const& child) {
    info[parent].children.push_back(child.top_node());
    for(auto const& item: child.info) {
      info.insert(item);
    }
  }

  void insert_child(int parent, tree_t<T> && child) {
    info[parent].children.push_back(child.top_node());
    info.merge(child.info);
  }

  int top_node() const {
    return _top_node;
  }

  bool is_leaf(int id) const {
    return info[id].children.size() > 0;
  }

private:
  struct info_t {
    T item;
    vector<int> children;
  };
  unordered_map<int, info_t> info;

  int _top_node;
};

// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

template<typename T>
struct is_vector {
  static const bool value=false;
};

template<typename T>
struct is_vector<std::vector<T>> {
  static const bool value=true;
};

template <typename T>
struct hash_combine_t {
  std::size_t operator()(tuple<int, T> const& x) const {
    auto const& [a,b] = x;
    std::hash<int> hasher;
    std::size_t ret = hasher(a);
    if constexpr (is_vector<T>::value) {
      for(auto const& bb: b) {
        hash_combine(ret, bb);
      }
    } else {
      hash_combine(ret, b);
    }
    return ret;
  }
};

// Every node n in a tree is going to be assigned a value from options[n].
// The cost associatiad with a particular assignment is the sum of
// (1) For every node n, f_cost_node(n, ret[n])
// (2) For every (parent p, child c), f_cost_edge(p, ret[p], c, ret[c]).
// This is a dynamic programming algorithm that finds the best placement across
// all possible assignments.

template <typename T>
using f_cost_node_t = std::function<uint64_t(int, T const&)>;

template <typename T>
using f_cost_edge_t = std::function<uint64_t(int, T const&,
                                             int, T const&)>;

template <typename T, typename Hash = hash_combine_t<T> >
tree_t<T> solve(
  tree_t<vector<T>> const& options,
  f_cost_node_t<T> f_cost_node,
  f_cost_edge_t<T> f_cost_edge)
{
  std::unordered_map<
    tuple<int, T>,
    tuple<uint64_t, tree_t<T> >,
    Hash
  > cache;

  using iterator_t = decltype(cache.find(tuple<int, T>()));

  auto update_cache = [&](int parent, T const& parent_item, int id) {
    // Compute the best cost and with what children of state
    uint64_t best_cost = std::numeric_limits<uint64_t>::max();
    vector<iterator_t> best_iters;
    T const* best_item = nullptr;

    for(auto const& item: options[id]) {
      uint64_t total = f_cost_node(id, item) + f_cost_edge(parent, parent_item, id, item);
      vector<iterator_t> iters;
      for(auto const& child: options.children(id)) {
        iters.push_back(cache.find({child, item}));
        auto const& [_0, _1] = *iters.back();
        auto const& [c, _2] = _1;
        total += c;
      }

      if(total < best_cost) {
        best_cost  = total;
        best_iters = iters;
        best_item  = &item;
      }
    }

    // Now create a tree and insert it into the cache

    tree_t<T> tree;
    tree.insert_root(id, *best_item);
    for(auto& iter: best_iters) {
      auto const& [_0, _1] = *iter;
      auto const& [_2, child_tree] = _1;
      tree.insert_child(id, child_tree);
    }

    cache.insert({
      tuple<int, T>(id, parent_item),
      tuple<uint64_t, tree_t<T>>(best_cost, std::move(tree))
    });
  };

  // Get a depth first ordering of the tree
  vector<int> ids;
  ids.push_back(options.top_node());
  for(int idx = 0; idx != ids.size(); ++idx) {
    for(auto const& child: options.children(ids[idx])) {
      ids.push_back(child);
    }
  }

  // Going bottom up, cache the info.
  //   parent ->  id  -> child
  // At each iteration,
  //  (1) (id, parent_item) is cached using (child, id_item) info and then
  //  (2) (child, id_item) is removed from the cache since it is unneeded.
  for(int idx = ids.size() - 1; idx >= 0; --idx) {
    int parent = ids[idx];

    for(auto const& parent_item: options[parent]) {
      for(auto const& id: options.children(parent)) {
        update_cache(parent, parent_item, id);
      }
    }

    // TODO: Figure out when things can be erased...It turns out for a use
    //       case, this code too eagerly erased something
    // // Since this is a tree, we know that parent's children will not
    // // be used again.
    // for(auto const& id: options.children(parent)) {
    //   for(auto const& item: options[parent]) {
    //     for(auto const& children: options.children(id)) {
    //       cache.erase(tuple<int, T>(children, item));
    //     }
    //   }
    // }
  }

  // At this point, we can find the best item for the top node.
  int top = ids[0];

  uint64_t best_cost = std::numeric_limits<uint64_t>::max();
  vector<iterator_t> best_iters;
  T const* best_item = nullptr;

  for(auto const& item: options[top]) {
    uint64_t total = f_cost_node(top, item);
    vector<iterator_t> iters;
    for(auto const& child: options.children(top)) {
      iters.push_back(cache.find({child, item}));
      auto const& [_0, _1] = *iters.back();
      auto const& [c, _2] = _1;
      total += c;
    }

    if(total < best_cost) {
      best_cost  = total;
      best_iters = iters;
      best_item  = &item;
    }
  }

  tree_t<T> ret;
  ret.insert_root(top, *best_item);
  for(auto& iter: best_iters) {
    auto const& [_0, _1] = *iter;
    auto const& [_2, child_tree] = _1;
    ret.insert_child(top, child_tree);
  }

  return ret;
}

//int main() {
//  vector<int> options{1,2,3,5,6,7,8,9,10,11,12};
//
//  tree_t<vector<int>> t;
//
//  t.insert_root(0, options);
//
//  t.insert_child(0, 1,  options);
//  t.insert_child(0, 2,  options);
//
//  t.insert_child(1, 3,  options);
//  t.insert_child(1, 4,  options);
//
//  t.insert_child(2, 5,  options);
//  t.insert_child(2, 6,  options);
//
//  t.insert_child(4, 7,  options);
//  t.insert_child(4, 8,  options);
//
//  t.insert_child(8, 9,  options);
//
//  t.insert_child(9, 10, options);
//
//  function<uint64_t(int, int const&)> f_cost_node =
//    [](int, int const& x) -> uint64_t
//  {
//    if(x > 5) {
//      return 0;
//    }
//    return x;
//  };
//
//  function<uint64_t(int, int const&,
//                    int, int const&)> f_cost_edge =
//    [](int, int const& x, int, int const& y) -> uint64_t
//  {
//    if(x != y) {
//      return 0;
//    }
//    return 1000;
//  };
//
//  tree_t<int> solved = solve(t, f_cost_node, f_cost_edge);
//
//  for(int i = 0; i != 11; ++i) {
//    std::cout << i << ": " << solved[i] << std::endl;
//  }
//}

}

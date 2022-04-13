#include "dag.h"
#include <cassert>
#include <algorithm>
#include <queue>

#define DCB01(x)          // std::cout << "dag.cc " << __LINE__ << " " << x << std::endl;

namespace bbts { namespace dag {

using std::function;

bbts::command_param_t to_bbts_param(param_t p) {
  if(p.which == param_t::which_t::F) {
    return { .f = p.val.f };
  }

  if(p.which == param_t::which_t::B) {
    return { .b = p.val.b };
  }

  if(p.which == param_t::which_t::I) {
    // TODO: die violently if the integer is too big
    using unsigned_integer_type = decltype(bbts::command_param_t().u);
    assert(p.val.i >= 0);
    return { .u = static_cast<unsigned_integer_type>(p.val.i) };
  }

  throw std::runtime_error("should not reach: to_bbts_param");
  return bbts::command_param_t();
}

vector<bbts::command_param_t> node_t::get_bbts_params() const {
  vector<bbts::command_param_t> ret;
  ret.reserve(params.size());
  for(auto const& p: params) {
    ret.push_back(to_bbts_param(p));
  }
  return ret;
}

std::ostream& node_t::print(std::ostream& os) const
{
  auto print_params = [this, &os]() {
    os << "[";
    for(param_t const& p: params) {
      os << p;
    }
    os << "]";
  };

  switch(type) {
    case node_type::input:
      os << "I";
      print_params();
      break;
    case node_type::reblock:
      os << "R";
      print_params();
      os << downs[0];
      break;
    case node_type::join:
      os << "J";
      print_params();
      for(int i = 0; i != downs.size(); ++i) {
        if(i > 0) {
          os << "$";
        }
        os << downs[i];
        for(auto const& rank: ordering[i]) {
          os << "," << rank;
        }
      }
      os << ":";
      print_list(os, aggs);
      break;
    case node_type::agg:
      os << "A";
      print_params();
      os << downs[0];
      break;

  }
  os << "|";
  print_list(os, dims);
  return os;
}

std::ostream& operator<<(std::ostream& os, param_t p)
{
  if(p.which == param_t::which_t::I) {
    os << "i" << p.val.i;
    return os;
  }
  if(p.which == param_t::which_t::F) {
    os << "f" << p.val.f;
    return os;
  }
  if(p.which == param_t::which_t::B) {
    os << "b";
    if(p.val.b) {
      os << "1";
    } else {
      os << "0";
    }
    return os;
  }
  throw std::runtime_error("should not reach here: operator << for param_t");
  return os;
}

std::istream& operator>>(std::istream& is, param_t& p) {
  char c;
  is >> c;
  if(c == 'i') {
    p.which = param_t::which_t::I;
    is >> p.val.i;
    return is;
  }
  if(c == 'f') {
    p.which = param_t::which_t::F;
    is >> p.val.f;
    return is;
  }
  if(c == 'b') {
    p.which = param_t::which_t::B;
    char b;
    is >> b;
    if(b == '0') {
      p.val.b = false;
    } else if(b == '1') {
      p.val.b = true;
    } else {
      throw std::runtime_error("should not reach here: operator >> for param_t, b val");
    }
    return is;
  }
  throw std::runtime_error("should not reach here: operator >> for param_t");
  return is;
}

vector<nid_t> dag_t::_set_inputs() {
  vector<nid_t> ret;
  for(node_t const& node: dag) {
    if(node.type == node_t::node_type::input) {
      ret.push_back(node.id);
    }
  }
  return ret;
}

vector<nid_t> dag_t::_set_breadth_dag_order() {
  vector<nid_t> counts(dag.size());
  vector<nid_t> pending;

  for(nid_t id = 0; id != dag.size(); ++id) {
    counts[id] = dag[id].downs.size();
    if(counts[id] == 0) {
      pending.push_back(id);
    }
  }

  vector<nid_t> ret;
  auto add_to_ret = [&](nid_t id) {
    ret.push_back(id);
    for(nid_t const& up: dag[id].ups) {
      counts[up]--;
      if(counts[up] == 0) {
        pending.push_back(up);
      }
    }
  };

  for(int idx = 0; idx != pending.size(); ++idx) {
    add_to_ret(pending[idx]);
  }

  return ret;
}

vector<nid_t> dag_t::_set_depth_dag_order() {
  vector<nid_t> counts(dag.size());
  vector<nid_t> inputs;
  for(nid_t id = 0; id != dag.size(); ++id) {
    counts[id] = dag[id].downs.size();
    if(counts[id] == 0) {
      inputs.push_back(id);
    }
  }

  vector<nid_t> ret;

  for(nid_t const& input: inputs) {
    // recursively add everything
    _depth_dag_order_add_to_ret(counts, ret, input);
  }

  return ret;
}

void dag_t::_depth_dag_order_add_to_ret(
  vector<nid_t>& counts,
  vector<nid_t>& ret,
  nid_t id) const
{
  ret.push_back(id);
  for(nid_t const& up: dag[id].ups) {
    counts[up]--;
    if(counts[up] == 0) {
      _depth_dag_order_add_to_ret(counts, ret, up);
    }
  }
}

struct priority_compare_t {
  priority_compare_t(function<int(nid_t)> f): f(f) {}

  bool operator()(nid_t const& lhs, nid_t const& rhs) {
    return f(lhs) > f(rhs);
  }

  function<int(nid_t)> f;
};

vector<nid_t> dag_t::priority_dag_order(
  function<int(nid_t)> get_priority) const
{
  using queue_t = std::priority_queue<nid_t, vector<nid_t>, priority_compare_t>;

  queue_t queue = queue_t(priority_compare_t(get_priority));

  vector<nid_t> ret;
  ret.reserve(dag.size());

  vector<int> counts;
  counts.reserve(dag.size());
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    counts[nid] = dag[nid].downs.size();
  }

  for(nid_t const& input: inputs()) {
    queue.push(input);
  }

  while(!queue.empty()) {
    nid_t cur = queue.top();
    queue.pop();

    ret.push_back(cur);

    for(nid_t up: dag[cur].ups) {
      counts[up]--;
      if(counts[up] == 0) {
        queue.push(up);
      }
    }
  }

  return ret;
}

vector<nid_t> dag_t::get_part_owners() const {
  vector<nid_t> ret;
  ret.reserve(dag.size());
  for(node_t const& node: dag) {
    if(node.is_part_owner()) {
      ret.push_back(node.id);
    }
  }
  return ret;
}

nid_t dag_t::get_part_owner(nid_t nid) const {
  node_t const& node = dag[nid];
  if(node.is_part_owner()) {
    return nid;
  }

  if(node.type == node_t::node_type::reblock) {
    assert(node.ups.size() == 1);
    node_t const& join_node = dag[node.ups[0]];
    assert(join_node.type == node_t::node_type::join);
    return join_node.id;
  }

  if(node.type == node_t::node_type::agg) {
    assert(node.downs.size() == 1);
    node_t const& join_node = dag[node.downs[0]];
    assert(join_node.type == node_t::node_type::join);
    return join_node.id;
  }

  assert(false);
  return 0;
}

vector<int> dag_t::get_out(vector<int> const& xs, nid_t nid) const {
  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::join) {
    vector<int> ret;
    for(int which_inc = 0; which_inc != xs.size(); ++which_inc) {
      auto iter = std::find(node.aggs.begin(), node.aggs.end(), which_inc);
      bool is_agg_dim = iter != node.aggs.end();
      if(!is_agg_dim) {
        ret.push_back(xs[which_inc]);
      }
    }
    return ret;
  } else {
    return xs;
  }
}

vector<int> dag_t::get_agg(vector<int> const& xs, nid_t nid) const {
  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::join) {
    vector<int> ret;
    for(int which_inc = 0; which_inc != xs.size(); ++which_inc) {
      auto iter = std::find(node.aggs.begin(), node.aggs.end(), which_inc);
      bool is_agg_dim = iter != node.aggs.end();
      if(is_agg_dim) {
        ret.push_back(xs[which_inc]);
      }
    }
    return ret;
  } else {
    return xs;
  }
}

vector<int> dag_t::get_out_for_input(
  vector<int> const& up_inc,
  nid_t upp_id,
  nid_t dwn_id) const
{
  node_t const& upp = dag[upp_id];
  node_t const& dwn = dag[dwn_id];

  // agg and reblock nodes have the same input shapes as outputs
  if(upp.type == node_t::node_type::agg) {
    return up_inc;
  } else
  if(upp.type == node_t::node_type::reblock) {
    return up_inc;
  } else
  if(upp.type == node_t::node_type::join) {
    // continue on
  } else {
    assert(false);
    return {};
  }

  node_t const& join_node = upp;
  vector<int> const& join_inc = up_inc;

  int which_input = 0;
  for(; which_input != join_node.downs.size(); ++which_input) {
    if(join_node.downs[which_input] == dwn_id) {
      break;
    }
  }
  if(which_input == join_node.downs.size()) {
    throw std::runtime_error("up join node doesn't have correct down");
  }

  assert(join_node.ordering.size() == join_node.downs.size());
  auto const& ordering = join_node.ordering[which_input];

  assert(ordering.size() == dwn.dims.size());

  vector<dim_t> ret;
  for(auto which_inc: ordering) {
    ret.push_back(join_inc[which_inc]);
  }
  return ret;
}

vector<int> dag_t::get_out_from_owner_incident(
  vector<int> const& inc,
  nid_t nid) const
{
  node_t const& node = dag[nid];
  if(node.type == node_t::node_type::reblock) {
    return get_out_for_input(inc, node.ups[0], nid);
  }
  if(node.type == node_t::node_type::agg) {
    return get_out(inc, node.downs[0]);
  }
  if(node.type == node_t::node_type::join) {
    return get_out(inc, nid);
  }
  if(node.type == node_t::node_type::input) {
    return inc;
  }
  throw std::runtime_error("get_out_from_owner_incident: invalid node type");
  return {};
}

vector<int> dag_t::get_node_incident(vector<int> const& inc, nid_t nid) const {
  nid_t owner_nid = get_part_owner(nid);
  if(owner_nid == nid) {
    return inc;
  }

  node_t const& node = dag[nid];

  if(node.type == node_t::node_type::reblock) {
    return get_out_for_input(inc, node.ups[0], nid);
  }

  if(node.type == node_t::node_type::agg) {
    return get_out(inc, owner_nid);
  }

  throw std::runtime_error("should not reach");
  return {};
}

vector<int> dag_t::combine_out_agg(
  vector<int> const& which_aggs,
  vector<int> const& out,
  vector<int> const& agg)
{
  assert(agg.size() == which_aggs.size());

  auto is_agg = [&which_aggs](int const& i) {
    return std::find(which_aggs.begin(), which_aggs.end(), i) != which_aggs.end();
  };

  vector<int> ret(out.size() + agg.size());

  auto out_iter = out.begin();
  auto agg_iter = agg.begin();
  for(int i = 0; i != ret.size(); ++i) {
    if(is_agg(i)) {
      ret[i] = *agg_iter++;
    } else {
      ret[i] = *out_iter++;
    }
  }

  return ret;
}

}}


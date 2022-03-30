#include "node.h"

namespace bbts { namespace dag {

std::ostream& node_t::print(std::ostream& os) const
{
  auto print_params = [this, &os]() {
    os << "[";
    for(param_t const& p: params) {
      std::cout << p;
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

}}

using namespace bbts::dag;

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
  throw std::runtime_error("should not reach here");
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
      throw std::runtime_error("should not reach here");
    }
    return is;
  }
  throw std::runtime_error("should not reach here");
  return is;
}


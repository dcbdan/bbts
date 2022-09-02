#include "parse.h"

namespace bbts { namespace dag {

enum token_t {
  t_node,
  t_bar,
  t_comma,
  t_uint,
  t_dollar,
  t_newline,
  t_colon,
  t_params,
  t_done
};

// Just for debugging
std::ostream& operator<<(std::ostream& os, token_t t) {
  switch(t) {
    case t_node:     os << "t_node";        break;
    case t_bar:      os << "t_bar";         break;
    case t_comma:    os << "t_comma";       break;
    case t_uint:     os << "t_uint";        break;
    case t_dollar:   os << "t_dollar";      break;
    case t_newline:  os << "t_newline";     break;
    case t_colon:    os << "t_colon";       break;
    case t_done:     os << "t_done";        break;
  }
  return os;
}

struct tokenizer_t {
  tokenizer_t(std::string filename, bool discard_first_line = true): f(filename)
  {
    if(discard_first_line) {
      std::string _discard;
      std::getline(f, _discard);
    }
  }

  token_t operator()() {
    if(f.eof()) {
      return t_done;
    }

    char c = f.peek();
    if(std::isdigit(static_cast<unsigned char>(c))) {
      f >> i;
      return t_uint;
    } else {
      f.get();
      switch(c) {
        case std::ifstream::traits_type::eof():
          return t_done;
        case '\n':
          return t_newline;
        case 'I':
          n = node_t::node_type::input;
          return t_node;
        case 'R':
          n = node_t::node_type::reblock;
          return t_node;
        case 'J':
          n = node_t::node_type::join;
          return t_node;
        case 'A':
          n = node_t::node_type::agg;
          return t_node;
        case 'M':
          n = node_t::node_type::mergesplit;
          return t_node;
        case '$':
          return t_dollar;
        case '|':
          return t_bar;
        case ',':
          return t_comma;
        case ':':
          return t_colon;
        case '[':
          parse_params();
          return t_params;
        default:
          die("operator() " + std::string(1,c));
      }
    }
    // should never get here
    die("operator() reached");
    return t_newline;
  }

  token_t operator()(std::vector<token_t> const& es) {
    token_t got = this->operator()();
    for(token_t const& e: es) {
      if(e == got) {
        return got;
      }
    }
    die("operator unexpected");
    return got;
  }

  void expect(token_t expect) {
    this->operator()({expect});
  }

  void die(std::string s) {
    throw std::runtime_error(
      "Sweet pony of Sierra Leone! The tokenizer_t is keeled over. " + s);
  }

  size_t i;
  node_t::node_type n;
  std::vector<param_t> ps;

private:
  std::ifstream f;

  void parse_params() {
    ps.resize(0);
    param_t p;
    while(f.peek() != ']') {
      f >> p;
      ps.push_back(p);
    }
    f.get();
  }
};

std::vector<std::vector<int> > read_list_of_nonempty_list(
  tokenizer_t& tk,
  token_t stop_inner,
  token_t stop)
{
  token_t t;
  std::vector<std::vector<int> > ret;
  std::vector<int> cur;

  t = tk({t_uint, stop});
  while(t != stop) {
    cur.push_back(tk.i);
    t = tk({t_comma, stop_inner, stop});
    while(t == t_comma) {
      tk.expect(t_uint);
      cur.push_back(tk.i);
      t = tk({t_comma, stop_inner, stop});
    }
    ret.push_back(cur);

    if(t == stop_inner) {
      cur.resize(0);
      tk.expect(t_uint);
    }
  }
  return ret;
}

std::vector<int> read_list(tokenizer_t& tk, token_t stop)
{
  token_t t;
  std::vector<int> ret;

  t = tk({t_uint, stop});
  if(t == stop) {
    return ret;
  } else {
    ret.push_back(tk.i);
  }
  while(true) {
    t = tk({t_comma, stop});
    if(t == stop) {
      return ret;
    }
    tk.expect(t_uint);
    ret.push_back(tk.i);
  }
}

// Matrix multiply example:
//   <first line is discarded>
//   I[i<ud>...ud params...]|40,50
//   I[i<ud>...ud params...]|50,60
//   R0[i<ud>...ud params...]|40,50
//   R1[i<ud>...ud params...]|50,60
//   J2[i<ud>...ud params...],0,1$3,1,2:1|40,50,60
//   A4[i<ud>...ud params...]|40,60
//
// MergeSplit node
// M[i<split size for leading dim or 0 if merge>]...
void parse_dag_into(std::string filename, std::vector<node_t>& ret) {
  tokenizer_t tk(filename);

  while(true) {
    token_t t = tk({t_node, t_done});
    if(t == t_done){
      // Before returning the dag, add all of the up values
      // and all of the id values
      for(int u = 0; u != ret.size(); ++u) {
        ret[u].id = u;

        for(auto const& d: ret[u].downs) {
          ret[d].ups.push_back(u);
        }
      }
      return;
    }
    node_t n;
    n.type = tk.n;

    // copy the kernel id (if join node) and the params (all nodes) over
    tk.expect(t_params);

    int start_ps, end_ps;
    if(n.type == node_t::node_type::join) {
      // The first param of a join node better be an int
      n.join_kernel = static_cast<node_t::join_kernel_type>(tk.ps[0].get_int());

      start_ps = 1;
      end_ps = tk.ps.size();
    } else if(n.type == node_t::node_type::mergesplit) {
      n.is_merge = tk.ps[0].get_int() == 0;
      start_ps = 1;
      end_ps = tk.ps.size();
    } else {
      start_ps = 0;
      end_ps = tk.ps.size();
    }

    // add the params
    n.params.reserve(end_ps - start_ps);
    for(int i = start_ps; i != end_ps; ++i) {
      n.params.push_back(tk.ps[i]);
    }

    switch(n.type) {
      case node_t::node_type::input:
        tk.expect(t_bar);
        break;
      case node_t::node_type::reblock:
        tk.expect(t_uint);
        n.downs.push_back(tk.i);
        tk.expect(t_bar);
        break;
      case node_t::node_type::join:
        {
          auto child_orderings = read_list_of_nonempty_list(
                                    tk, t_dollar, t_colon);
          for(auto v: child_orderings) {
            n.downs.push_back(v[0]);
            n.ordering.emplace_back(v.begin() + 1, v.end());
          }
        }
        n.aggs = read_list(tk, t_bar);
        break;
      case node_t::node_type::agg:
        tk.expect(t_uint);
        n.downs.push_back(tk.i);
        tk.expect(t_bar);
        break;
      case node_t::node_type::mergesplit:
        tk.expect(t_uint);
        n.downs.push_back(tk.i);
        tk.expect(t_bar);
        break;
    }

    n.dims = read_list(tk, t_newline);
    ret.push_back(n);
  }
}

std::vector<node_t> parse_dag(std::string filename) {
  std::vector<node_t> ret;
  parse_dag_into(filename, ret);
  return ret;
}

void print_dag(std::vector<node_t> const& dag) {
  for(auto const& node: dag) {
    std::cout << node << "\n";
  }
}

}}

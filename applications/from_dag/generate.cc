#include "generate.h"
#include "misc.h"
#include "../../src/utils/expand_indexer.h"

#include <algorithm>
#include <numeric>
#include <map>

#define DCB01(x) // std::cout << "generate " << __LINE__ << " " << x << std::endl
#define HAS_PERMUTE(x) // std::cout << "HAS PERMUTE " << x << std::endl

namespace bbts { namespace dag {

using utils::expand::expand_indexer_t;
using utils::expand::column_major_expand_t;

generate_commands_t::generate_commands_t(
  dag_t const& dag_,
  relations_t const& relations_,
  vector<vector<int>> const& compute_locs_,
  ud_info_t ud_info_,
  int num_nodes_):
    dag(dag_),
    relations(relations_),
    compute_locs(compute_locs_),
    ud_info(ud_info_),
    num_nodes(num_nodes_),
    _command_id(0),
    _tid(0)
{
  // Setup the information that the relations index
  tid_locs.reserve(dag.size());
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    tid_locs.push_back(
      vector<tid_loc_t>(
        relations[nid].get_num_blocks(),
        {-1,-1}));
  }

  // Add the nodes
  for(nid_t which: dag.breadth_dag_order()) {
    add_node(which);
  }

  // And now the deletes
  add_deletes();
}

////////////////////////////////////////////////////////////////////////////////

// The type system here is not as explicit as the haskell code below...
//
// Here, a permutation of size n is a vector of size n that has values 0,1,...,n-1 in it.
// permute and inverse `ps` argument is a permutation.
// inverse and from_to returns a permutation.

bool is_permutation(vector<int> ps) {
  vector<int> xs;
  xs.reserve(ps.size());
  for(int i = 0; i != ps.size(); ++i) {
    xs.push_back(i);
  }

  std::sort(ps.begin(), ps.end());

  return ps == xs;
}

template <typename T>
vector<T> permute(vector<int> const& ps, vector<T> const& xs) {
  assert(is_permutation(ps));

  vector<T> ret(ps.size());
  for(int i = 0; i != ps.size(); ++i) {
    ret[i] = xs[ps[i]];
  }
  return ret;
}

vector<int> inverse(vector<int> const& ps) {
  assert(is_permutation(ps));

  vector<int> ret(ps.size());
  for(int i = 0; i != ps.size(); ++i) {
    ret[ps[i]] = i;
  }
  return ret;
}

vector<int> from_to(vector<int> const& inn, vector<int> out) {
  // inn and out need not actually be permutations from
  // [0,..(n-1)] and can insead be over other domains,
  // but the output should be a proper permutation

  std::map<int,int> relabel;
  int w = 0;
  for(auto const& i: inn) {
    relabel[i] = w++;
  }
  for(int& i: out) {
    i = relabel[i];
  }

  assert(is_permutation(out));
  return out;
}

//import qualified Data.List as List
//
//data Perm = Perm [Int] deriving (Show, Eq)
//
//permute :: Ord a => Perm -> [a] -> [a]
//permute (Perm ps) xs = map (xs!!) ps
//
//permute_also (Perm ps) xs = out
//  where out = map snd $ List.sort $ zip ps xs
//
//inverse :: Perm -> Perm
//inverse (Perm ps) = Perm inv
//  where inv = map snd $ List.sort $ zip ps [0..]
//
//from_to :: Perm -> Perm -> Perm
//from_to (Perm inns) (Perm outs) = Perm ret
//  where ret = map snd $ List.sort $ zip inns outs
//
//main :: IO ()
//main = do
//  print $ (permute (Perm [2,3,0,1]) [0,1,2,3]    == [2,3,0,1])
//  print $ (inverse (Perm [2,3,0,1])              == Perm [2,3,0,1])
//  print $ (inverse (Perm [2,0,1,3])              == Perm [1,2,0,3])
//  print $ (from_to (Perm [0,1,2]) (Perm [2,1,0]) == Perm [2,1,0])
//  print $ (from_to (Perm [2,0,1]) (Perm [0,1,2]) == inverse (Perm [2,0,1]))
//
//int main() {
//  {
//    vector<int> p = {2,3,0,1};
//    vector<int> x = {0,1,2,3};
//    std::cout << std::boolalpha << (permute(p, x) == p) << std::endl;
//  }
//  {
//    vector<int> p = {2,3,0,1};
//    std::cout << std::boolalpha << (inverse(p) == p) << std::endl;
//  }
//  {
//    vector<int> x = {2,0,1,3};
//    vector<int> y = {1,2,0,3};
//    std::cout << std::boolalpha << (inverse(x) == y) << std::endl;
//  }
//  {
//    vector<int> x = {0,1,2};
//    vector<int> y = {2,1,0};
//    std::cout << std::boolalpha << (from_to(x,y) == y) << std::endl;
//  }
//  {
//    vector<int> x = {2,0,1};
//    vector<int> y = {0,1,2};
//    std::cout << std::boolalpha << (from_to(x,y) == inverse(x)) << std::endl;
//  }
//  {
//    vector<int> x = {0,3,1,4,2};
//    vector<int> y = {9,10,11,12,13};
//    std::cout << std::boolalpha
//      << (permute(inverse(x), permute(x, y)) == y)
//      << std::endl;
//  }
//}
////////////////////////////////////////////////////////////////////////////////

// A join node may request a contraction, elementwise, elementwise binary
// or a reduction. In general, the join node allows arbitrary permutations
// of the outputs. But the corresponding operations that actually do the
// implementations have various ordering constraints on the ranks. To
// remedy this, permutations may have to be  applied to the inputs of the
// join op or to the output.
//
// This handler computes all of those requirements and figures out what
// the necessary params will be. Then generate_commands_t::add_node uses
// that info to generate the necessary commands.

class join_handler_t {
  vector<tuple<int, vector<bbts::command_param_t>>> _input_permutes;
  vector<bbts::command_param_t> _op_params;

  bool _post_permute;
  vector<bbts::command_param_t> _post_permute_params;

  bool _is_no_op_unary_ew;

  struct einsum_order_t {
    einsum_order_t(
      vector<int> const& modes_lhs,
      vector<int> const& modes_rhs,
      vector<int> const& modes_out)
    {
      for(auto const& m: modes_out) {
        bool in_lhs = std::find(modes_lhs.begin(), modes_lhs.end(), m) != modes_lhs.end();
        bool in_rhs = std::find(modes_rhs.begin(), modes_rhs.end(), m) != modes_rhs.end();
        if(in_lhs && in_rhs) {
          bs.push_back(m);
        } else
        if(in_lhs) {
          is.push_back(m);
        } else
        if(in_rhs) {
          ks.push_back(m);
        } else {
          assert(false);
        }
      }
      for(auto const& m: modes_lhs) {
        bool in_rhs = std::find(modes_rhs.begin(), modes_rhs.end(), m) != modes_rhs.end();
        bool in_out = std::find(modes_out.begin(), modes_out.end(), m) != modes_out.end();
        if(in_rhs && !in_out) {
          js.push_back(m);
        }
      }
    }

    bool is_ijb(vector<int> const& modes) const {
      return is_(modes, is, js, bs);
    }
    bool is_jib(vector<int> const& modes) const {
      return is_(modes, js, is, bs);
    }
    bool is_jkb(vector<int> const& modes) const {
      return is_(modes, js, ks, bs);
    }
    bool is_kjb(vector<int> const& modes) const {
      return is_(modes, ks, js, bs);
    }
    bool is_ikb(vector<int> const& modes) const {
      return is_(modes, is, ks, bs);
    }

    vector<int> permute_to_ijb(vector<int> const& modes) const {
      return permute_to_(modes, is, js, bs);
    }
    vector<int> permute_to_jkb(vector<int> const& modes) const {
      return permute_to_(modes, js, ks, bs);
    }
    vector<int> permute_from_ikb(vector<int> const& modes) const {
      return permute_from_(is, ks, bs, modes);
    }

    int ni() const { return static_cast<int>(is.size()); }
    int nj() const { return static_cast<int>(js.size()); }
    int nk() const { return static_cast<int>(ks.size()); }
    int nb() const { return static_cast<int>(bs.size()); }

  private:
    vector<int> is, js, ks, bs;

    bool is_(
      vector<int> const& modes,
      vector<int> const& as,
      vector<int> const& bs,
      vector<int> const& cs) const
    {
      assert(modes.size() == (as.size() + bs.size() + cs.size()));

      int i = 0;
      for(int const& a: as) { if(modes[i++] != a) { return false; } }
      for(int const& b: bs) { if(modes[i++] != b) { return false; } }
      for(int const& c: cs) { if(modes[i++] != c) { return false; } }

      return true;
    }

    vector<int> permute_to_(
      vector<int> const& modes_inn,
      vector<int> const& as,
      vector<int> const& bs,
      vector<int> const& cs) const
    {
      vector<int> modes_out;

      modes_out.reserve(as.size() + bs.size() + cs.size());
      for(int const& a: as) { modes_out.push_back(a); }
      for(int const& b: bs) { modes_out.push_back(b); }
      for(int const& c: cs) { modes_out.push_back(c); }

      assert(modes_inn.size() == modes_out.size());

      return from_to(modes_inn, modes_out);
    }

    vector<int> permute_from_(
      vector<int> const& as,
      vector<int> const& bs,
      vector<int> const& cs,
      vector<int> const& modes_out) const
    {
      vector<int> modes_inn;

      modes_inn.reserve(as.size() + bs.size() + cs.size());
      for(int const& a: as) { modes_inn.push_back(a); }
      for(int const& b: bs) { modes_inn.push_back(b); }
      for(int const& c: cs) { modes_inn.push_back(c); }

      assert(modes_inn.size() == modes_out.size());

      return from_to(modes_inn, modes_out);
    }

  };

public:
  join_handler_t(
    node_t::join_kernel_type kernel,
    vector<param_t> const& params)
  {
    _is_no_op_unary_ew = false;

    if(kernel == node_t::join_kernel_type::contraction) {
      // From BuildDag:
      //   (nl,nr,no), left_modes, right_modes, out_modes, alpha
      // For Kernel Library
      //   t_lhs, t_rhs alpha ni, nj, nk, nb
      //                      ^ here, n = number of ranks, not dimensions size
      //
      // Turn an arbitrary einsum into
      //   ijb,jkb->ikb
      //   jib,jkb->ikb
      //   ijb,kjb->ikb
      //   jib,kjb->ikb

      int num_lhs = params[0].get_int();
      int num_rhs = params[1].get_int();
      int num_out = params[2].get_int();

      assert(params.size() == 4 + num_lhs + num_rhs + num_out);

      int beg_lhs = 3;
      int end_lhs = beg_lhs + num_lhs;

      int beg_rhs = end_lhs;
      int end_rhs = beg_rhs + num_rhs;

      int beg_out = end_rhs;
      int end_out = beg_out + num_out;

      vector<int> modes_lhs; modes_lhs.reserve(num_lhs);
      vector<int> modes_rhs; modes_rhs.reserve(num_rhs);
      vector<int> modes_out; modes_out.reserve(num_out);

      for(int i = beg_lhs; i != end_lhs; ++i) { modes_lhs.push_back(params[i].get_int()); }
      for(int i = beg_rhs; i != end_rhs; ++i) { modes_rhs.push_back(params[i].get_int()); }
      for(int i = beg_out; i != end_out; ++i) { modes_out.push_back(params[i].get_int()); }

      float alpha = params[end_out].get_float();

      einsum_order_t info(modes_lhs, modes_rhs, modes_out);

      // Is the lhs already in jib or ijb form? If not, convert it to ijb.
      // Is the rhs already in jkb or kjb form? If not, convert it to jkb.

      bool t_lhs;
      if(info.is_ijb(modes_lhs)) {
        t_lhs = false;
      } else if(info.is_jib(modes_lhs)) {
        t_lhs = true;
      } else {
        t_lhs = false;

        vector<bbts::command_param_t> perm;
        for(auto const& a: info.permute_to_ijb(modes_lhs)) {
          perm.push_back(bbts::command_param_t { .i = a });
        }

        HAS_PERMUTE("contraction input left rank " << perm.size());
        _input_permutes.emplace_back(0, perm);
      }

      bool t_rhs;
      if(info.is_jkb(modes_rhs)) {
        t_rhs = false;
      } else if(info.is_kjb(modes_rhs)) {
        t_rhs = true;
      } else {
        t_rhs = false;

        vector<bbts::command_param_t> perm;
        for(auto const& a: info.permute_to_jkb(modes_rhs)) {
          perm.push_back(bbts::command_param_t { .i = a });
        }

        HAS_PERMUTE("contraction input right rank " << perm.size());
        _input_permutes.emplace_back(1, perm);
      }

      // Add the params

      _op_params.push_back(bbts::command_param_t { .i = t_lhs });
      _op_params.push_back(bbts::command_param_t { .i = t_rhs });

      _op_params.push_back(bbts::command_param_t { .f = alpha });

      _op_params.push_back(bbts::command_param_t { .i = info.ni() });
      _op_params.push_back(bbts::command_param_t { .i = info.nj() });
      _op_params.push_back(bbts::command_param_t { .i = info.nk() });
      _op_params.push_back(bbts::command_param_t { .i = info.nb() });

      // Are the output modes already in ikb form? If not, permute it.

      if(!info.is_ikb(modes_out)) {
        for(auto const& a: info.permute_from_ikb(modes_out)) {
          _post_permute_params.push_back(bbts::command_param_t { .i = a });
        }
        HAS_PERMUTE("contraction post rank " << _post_permute_params.size());
        _post_permute = true;
      } else {
        _post_permute = false;
      }

    } else
    if(kernel == node_t::join_kernel_type::reduction) {
      // ij->j
      //
      // From BuildDag:
      //   binary op, alpha, num input rank, out modes
      // For Kernel Library
      //   binary op, alpha, num modes i, num modes j

      int input_rank = params[2].get_int();
      vector<int> out_modes;
      out_modes.reserve(input_rank);
      for(int i = 3; i < params.size(); ++i) {
        out_modes.push_back(params[i].get_int());
      }

      vector<int> agg_modes;
      agg_modes.reserve(input_rank);
      for(int i = 0; i != input_rank; ++i) {
        if(std::find(out_modes.begin(), out_modes.end(), i) == out_modes.end()) {
          agg_modes.push_back(i);
        }
      }
      int num_agg = static_cast<int>(agg_modes.size());
      int num_out = static_cast<int>(out_modes.size());

      assert(num_agg + num_out == input_rank);

      _op_params.push_back(bbts::command_param_t {
        .i = params[0].get_int()
      });
      _op_params.push_back(bbts::command_param_t {
        .f = params[1].get_float()
      });
      _op_params.push_back(bbts::command_param_t {
        .i = num_agg
      });
      _op_params.push_back(bbts::command_param_t {
        .i = num_out
      });

      bool agg_then_out;
      if(out_modes.size() == 0) {
        agg_then_out = true;
      } else {
        int min_out = *std::min_element(out_modes.begin(), out_modes.end());
        int max_agg = *std::max_element(agg_modes.begin(), agg_modes.end());
        agg_then_out = min_out > max_agg;
      }

      vector<int> out_modes_after;
      if(agg_then_out)
      {
        // In this case, do not permute the input since all the aggs already
        // come first.

        // But do the output modes have to be permuted?
        // If the out modes are sorted, then the agg, out modes was of the form
        // (say [0,1,2], [3,4,5]) and already in the normal form.
        if(std::is_sorted(out_modes.begin(), out_modes.end())) {
          _post_permute = false;
        } else {
          _post_permute = true;

          // Example: agg_modes = [0,1,2], out_modes = [3,5,4]
          // The output modes aren't ascending and contigous, and so need to be
          // fixed..First, subtract 3 -> [0,2,1] and that is the permutation

          _post_permute_params.reserve(out_modes.size());
          for(auto const& o: out_modes) {
            _post_permute_params.push_back(bbts::command_param_t {
              .i = o - num_agg
            });
          }
          HAS_PERMUTE("reduction post rank " << _post_permute_params.size());
        }
      } else {
        // In this case, permute it so that the aggs come first and then the out
        // modes in the order requested.
        vector<bbts::command_param_t> perm;
        perm.reserve(agg_modes.size() + out_modes.size());
        for(auto const& a: agg_modes) {
          perm.push_back(bbts::command_param_t { .i = a });
        }
        for(auto const& o: out_modes) {
          perm.push_back(bbts::command_param_t { .i = o });
        }

        HAS_PERMUTE("reduction input rank " << perm.size());
        _input_permutes.emplace_back(0, perm);

        _post_permute = false;
      }

      // 1. Are t? If so, don't permute anything
      //    and reduce. Otherwise, permute so that i,j,b.
      // 2. Afterwards, check if the output ordering is correct. If so,
      //    we are done.
    } else
    if(kernel == node_t::join_kernel_type::unary_elementwise) {
      // From BuildDag:
      //   uop (which may be one or two params), alpha, out modes
      // For Kernel Library:
      //   uop (one or two params), alpha
      //
      // If the permutation specified in out modes is not the
      // identitiy permutation, permute the input. (One could
      // also permute the output, it doesn't matter)
      int i = 0;
      _op_params.push_back(bbts::command_param_t {
          .i = params[i++].get_int()
      });
      if(_op_params.back().i == 6 ||
         _op_params.back().i == 7 ||
         _op_params.back().i == 9)
      {
        _op_params.push_back(bbts::command_param_t {
          .f = params[i++].get_float()
        });
      }
      if(_op_params.back().i == 8) {
        _is_no_op_unary_ew = true;
      }
      _op_params.push_back(bbts::command_param_t {
        .f = params[i++].get_float()
      });

      vector<bbts::command_param_t> out_perm;
      out_perm.reserve(params.size());
      while(i != params.size()) {
        out_perm.push_back(bbts::command_param_t {
          .i = params[i++].get_int()
        });
      }

      bool has_input_permute = false;
      for(int x = 0; x != out_perm.size(); ++x) {
        if(x != out_perm[x].i) {
          has_input_permute = true;
          break;
        }
      }

      if(has_input_permute) {
        HAS_PERMUTE("unary input rank " << out_perm.size());
        _input_permutes.push_back({ 0, out_perm });
      }

      _post_permute = false;

      if(_is_no_op_unary_ew) {
        if(_post_permute || !has_input_permute) {
          throw std::runtime_error("no op uneary ew ops must have a pre "
                                   "permute and no post permute");
        }
      }
    } else
    if(kernel == node_t::join_kernel_type::binary_elementwise) {
      // FromDag
      //   bop alpha (nLhs, lhsOrd) (nRhs, rhsOrd)
      // For Kernel Library
      //   bop alpha (lhsOrd sorted) (rhsOrd sorted)

      int n_lhs = params[2].get_int();  // 3.. 3,4,5.. start at 6.
      int beg_lhs = 3;
      int end_lhs = beg_lhs + n_lhs;

      int n_rhs = params[end_lhs].get_int();
      int beg_rhs = end_lhs + 1;
      int end_rhs = beg_rhs + n_rhs;

      assert(end_rhs == params.size());

      vector<int> lhs_ord;
      lhs_ord.reserve(n_lhs);
      for(int i = beg_lhs; i != end_lhs; ++i) {
        lhs_ord.push_back(params[i].get_int());
      }

      vector<int> rhs_ord;
      rhs_ord.reserve(n_rhs);
      for(int i = beg_rhs; i != end_rhs; ++i) {
        rhs_ord.push_back(params[i].get_int());
      }

      // From dag might give something like this:
      //   [0,1],   [0,1,2] -> [0,1,2]
      //   ^lhsOrd  ^rhsOrd
      // In this case, there is nothign to do.
      // But:
      //   [0,1],[2,0,1] -> [0,1,2]
      // requires permuting the second input.
      // It needs to permuted to [2,0,1]->[0,1,2].
      // The permute params are in the form of
      //                         [0,1,2]->[x,y,z]
      // where x,y,z cover 0,1,2 in some order.
      //
      // In other words:
      //   If lhsOrd or rhsOrd are not sorted, a permutation needs to happen.
      //   The permutation params are the inverse of the provided ord.

      if(!std::is_sorted(lhs_ord.begin(), lhs_ord.end())) {
        // add a permutation to the lhs

        vector<bbts::command_param_t> perm;
        perm.reserve(lhs_ord.size());
        for(auto which: inverse(lhs_ord)) {
          perm.push_back(bbts::command_param_t {
            .i = which
          });
        }

        HAS_PERMUTE("binary input left " << perm.size());
        _input_permutes.emplace_back(0, perm);

        std::sort(lhs_ord.begin(), lhs_ord.end());
      }

      if(!std::is_sorted(rhs_ord.begin(), rhs_ord.end())) {
        // add a permutation to the rhs

        vector<bbts::command_param_t> perm;
        perm.reserve(lhs_ord.size());
        for(auto which: inverse(rhs_ord)) {
          perm.push_back(bbts::command_param_t {
            .i = which
          });
        }

        HAS_PERMUTE("binary input right " << perm.size());
        _input_permutes.emplace_back(1, perm);

        std::sort(rhs_ord.begin(), rhs_ord.end());
      }

      _op_params.push_back(bbts::command_param_t {
        .i = params[0].get_int()
      });
      _op_params.push_back(bbts::command_param_t {
        .f = params[1].get_float()
      });

      for(auto const& x: lhs_ord) {
        _op_params.push_back(bbts::command_param_t {
          .i = x
        });
      }
      for(auto const& x: rhs_ord) {
        _op_params.push_back(bbts::command_param_t {
          .i = x
        });
      }

      _post_permute = false;

    } else {
      assert(false);
    }
  }

  vector<tuple<int, vector<bbts::command_param_t>>> const&
  input_permutes() const { return _input_permutes; }

  vector<bbts::command_param_t> const&
  op_params() const { return _op_params; }

  bool has_post_permute() const { return _post_permute; }

  bool is_just_permute() const { return _is_no_op_unary_ew; }

  vector<bbts::command_param_t> const&
  post_permute_params() const
  {
    assert(has_post_permute());
    return _post_permute_params;
  }
};

vector<tid_loc_t>
generate_commands_t::get_inputs(nid_t an_nid, vector<int> const& an_bid, int prefer_loc) const
{
  vector<tuple<nid_t, int>> const& items = relations[an_nid].get_inputs(an_bid);

  vector<tid_loc_t> ret;
  ret.reserve(items.size());
  for(auto const& [input_nid, input_idx]: items) {
    auto [tid,compute_loc] = tid_locs[input_nid][input_idx];
    if(was_moved_to(tid, prefer_loc)) {
      ret.push_back({ tid, prefer_loc });
    } else {
      ret.push_back({ tid, compute_loc});
    }
  }
  return ret;
}

tid_loc_t&
generate_commands_t::get_tid_loc(nid_t an_nid, vector<int> const& an_bid)
{
  auto [nid, idx] = relations[an_nid][an_bid];
  return tid_locs[nid][idx];
}

void generate_commands_t::add_node(nid_t nid) {
  // don't do anything if this is actually a no op!
  // get_inputs will reach past no ops.
  if(relations[nid].is_no_op()) {
    return;
  }

  node_t const& node = dag[nid];

  // Here are some things specific to reblocking, but don't need to be
  // created over and over for every block
  std::unique_ptr<expand_indexer_t> expand_indexer_ptr(nullptr);
  if(node.type == node_t::node_type::reblock) {
    auto const& inn_partition = relations[node.downs[0]].partition;
    auto const& out_partition = relations[nid].partition;

    expand_indexer_ptr = std::unique_ptr<expand_indexer_t>(
      new expand_indexer_t(
        inn_partition,
        out_partition));
  }

  // Turns out if we're gonna do a mergesplit, its basically a reblock
  if(node.type == node_t::node_type::mergesplit) {
    vector<int> inn_partition;
    vector<int> out_partition;

    auto const& input_nid = node.downs[0];
    if(node.is_merge) {
      // IJij -> Kk (= 1K1k)
      inn_partition = relations[input_nid].partition;
      out_partition = expand1(relations[nid].partition);
    } else {
      // Kk (= 1K1k) -> IJij
      inn_partition = expand1(relations[input_nid].partition);
      out_partition = relations[nid].partition;
    }

    expand_indexer_ptr = std::unique_ptr<expand_indexer_t>(
      new expand_indexer_t(
        inn_partition,
        out_partition));
  }

  // Here are some things specific to join, but don't need to be
  // created over and over for every block
  std::unique_ptr<join_handler_t> join_handler(nullptr);
  if(node.type == node_t::node_type::join) {
    join_handler = std::unique_ptr<join_handler_t>(
      new join_handler_t(node.join_kernel, node.params));
  }

  // for each bid, add the command(s) to get the output
  indexer_t indexer(relations[nid].partition);
  do {
    auto const& bid = indexer.idx;

    // The compute location was pre computed somehow; use that
    int compute_location = compute_locs[nid][relations[nid].bid_to_idx(bid)];

    vector<tid_loc_t> inputs = get_inputs(nid, bid, compute_location);

    // Now we have all the inputs, figure out what the node is and use that
    // to create the command and get a new tid.

    if(node.type == node_t::node_type::input)
    {
      tid_loc_t cur{ next_tid(), compute_location };

      auto params = node.get_bbts_params();
      // input kernels in libbarb_cutensor require
      // appending for each rank
      //   (which blk, local dim size, output dim size)
      // as parameters
      using uint_type = decltype(bbts::command_param_t().u);
      auto const& ps = relations[nid].partition;
      for(int r = 0; r != bid.size(); ++r) {
        params.push_back({ .u = static_cast<uint_type>(bid[r])               });
        params.push_back({ .u = static_cast<uint_type>(node.dims[r] / ps[r]) });
        params.push_back({ .u = static_cast<uint_type>(node.dims[r])         });
      }

      input_commands.emplace_back(command_t::create_apply(
        next_command_id(),
        ud_info.init,
        false,
        params,
        {},
        {cur}));

      get_tid_loc(nid, bid) = cur;
    }
    else if(node.type == node_t::node_type::join)
    {
      // for each input, move it to the location if necessary
      vector<tid_loc_t> join_inputs;
      join_inputs.reserve(inputs.size());
      for(auto const& [tid,loc]: inputs) {
        if(loc != compute_location) {
          // move it if it hasn't yet been moved
          assure_moved_to(commands, tid, loc, compute_location);
        }
        join_inputs.push_back({tid, compute_location});
      }

      // Now we must
      //   (a) optionally permute the inputs
      //   (b) do the relevant join computation
      //   (c) optionally permute the ouptut
      //   (d) delete anything that was permuted
      // But the join handler will tell us all that info

      for(auto [which_input, permute_params]: join_handler->input_permutes()) {
        tid_loc_t previous = join_inputs[which_input];

        join_inputs[which_input] = tid_loc_t{ next_tid(), compute_location };
        tid_loc_t const& now = join_inputs[which_input];

        commands.emplace_back(command_t::create_apply(
          next_command_id(),
          ud_info.permute,
          false,
          permute_params,
          {previous},
          {now}));
      }

      tid_loc_t after_op;
      if(join_handler->is_just_permute()) {
        after_op = join_inputs[0];
      } else {
        after_op = tid_loc_t{ next_tid(), compute_location };
        commands.emplace_back(command_t::create_apply(
          next_command_id(),
          ud_info.get_join_ud(node.join_kernel),
          false,
          join_handler->op_params(),
          join_inputs,
          {after_op}));
      }

      tid_loc_t cur;
      if(join_handler->has_post_permute()) {
        cur = tid_loc_t{ next_tid(), compute_location };

        commands.emplace_back(command_t::create_apply(
          next_command_id(),
          ud_info.permute,
          false,
          join_handler->post_permute_params(),
          {after_op},
          {cur}));
      } else {
        cur = after_op;
      }

      // now do the clean up
      // (if this whole op was just a permute, then don't do the cleanup;
      //  also only one of the pre or post permutes happened)
      if(!join_handler->is_just_permute()) {
        for(auto [which_input, permute_params]: join_handler->input_permutes()) {
          commands.emplace_back(command_t::create_delete(
            next_command_id(),
            {join_inputs[which_input]}));
        }
        if(join_handler->has_post_permute()) {
          commands.emplace_back(command_t::create_delete(
            next_command_id(),
            {after_op}));
        }
      }

      get_tid_loc(nid, bid) = cur;
    }
    // Create a reduce command
    else if(node.type == node_t::node_type::agg)
    {
      tid_loc_t cur{ next_tid(), compute_location };

      // If all the inputs are not at compute_location, this guy will not work.
      // To handle this, first move the first input to the compute_location.
      bool has_one_input_at_compute_loc = false;
      for(auto const& [_0, input_loc]: inputs) {
        if(input_loc == compute_location) {
          has_one_input_at_compute_loc = true;
          break;
        }
      }
      if(!has_one_input_at_compute_loc) {
        auto const& [input_tid, input_loc] = inputs[0];
        assure_moved_to(commands, input_tid, input_loc, compute_location);

        auto inputs_copy = inputs;
        inputs_copy[0] = tid_loc_t{ input_tid, compute_location };
        commands.emplace_back(command_t::create_reduce(
          next_command_id(),
          ud_info.castable_elementwise,
          false,
          node.get_bbts_params(),
          inputs_copy,
          cur));
      } else {
        commands.emplace_back(command_t::create_reduce(
          next_command_id(),
          ud_info.castable_elementwise,
          false,
          node.get_bbts_params(),
          inputs,
          cur));
      }

      get_tid_loc(nid, bid) = cur;
    }
    else if(node.type == node_t::node_type::reblock ||
            node.type == node_t::node_type::mergesplit)
    {
      expand_indexer_t const& expand_indexer = *expand_indexer_ptr;

      vector<int> inn_partition;
      vector<int> out_partition;
      vector<int> bid_fixed;       // bid_fixed is with respect to a reblock

      if(node.type == node_t::node_type::reblock) {
        inn_partition = relations[node.downs[0]].partition;
        out_partition = relations[nid].partition;
        bid_fixed = bid;
      } else {
        if(node.is_merge) {
          // IJij -> Kk (= 1K1k)
          inn_partition = relations[node.downs[0]].partition;
          out_partition = expand1(relations[nid].partition);
          bid_fixed = expand0(bid);
        } else {
          // Kk (= 1K1k) -> IJij
          inn_partition = expand1(relations[node.downs[0]].partition);
          out_partition = relations[nid].partition;
          bid_fixed = bid;
        }
      }

      // the same reblock params are used for each of the inputs
      vector<command_param_t> reblock_params;
      reblock_params.reserve(4*node.dims.size());
      auto insert = [&](int x) {
        reblock_params.push_back({ .u = static_cast<uint32_t>(x) });
      };
      auto const& partition_here = relations[nid].partition;
      auto const& partition_down = relations[node.downs[0]].partition;
      for(int r = 0; r != node.dims.size(); ++r) {
        insert(node.dims[r]);
        insert(inn_partition[r]);
        insert(out_partition[r]);
        insert(bid_fixed[r]);
      }

      tid_loc_t cur{ next_tid(), compute_location };

      for(int which_input = 0; which_input != inputs.size(); ++which_input) {
        auto [tid,loc] = inputs[which_input];

        if(was_moved_to(tid, compute_location)) {
          loc = compute_location;
        }

        column_major_expand_t expand(expand_indexer.get_expand_dim(
          node.dims,
          expand_indexer.get_which_input(bid_fixed, which_input),
          bid_fixed));

        bool compact_is_new = false;
        tid_t compact_tid = tid;
        if(!expand.is_compact_inn()) {
          compact_tid = next_tid();
          compact_is_new = true;

          // do the compact into compact_tid
          commands.emplace_back(command_t::create_compact(
            next_command_id(),
            ud_info.expand,
            false,
            which_input,
            reblock_params,
            {tid, loc},
            {compact_tid, loc}));
        }
        // if the compact input needs to be moved, move it
        if(loc != compute_location) {
          assure_moved_to(commands, compact_tid, loc, compute_location);
          if(compact_is_new) {
            commands.emplace_back(command_t::create_delete(
              next_command_id(),
              {tid_loc_t{compact_tid, loc}}));
          }
        }
        // now a compact item exists at the destination location,
        // so do the touch
        commands.emplace_back(command_t::create_touch(
          next_command_id(),
          ud_info.expand,
          false,
          which_input,
          inputs.size(),
          reblock_params,
          {tid_loc_t{compact_tid, compute_location}},
          cur));
        if(compact_is_new) {
          commands.emplace_back(command_t::create_delete(
            next_command_id(),
            {tid_loc_t{compact_tid, compute_location}}));
        }
      }

      // do _not_ used bid_fixed here! This assigns the output at bid
      get_tid_loc(nid, bid) = cur;
    } else {
      throw std::runtime_error("should not reach");
    }

  } while (indexer.increment());
}

tuple<vector<command_ptr_t>, vector<command_ptr_t>>
generate_commands_t::extract()
{
  tuple<vector<command_ptr_t>, vector<command_ptr_t>> ret{
    std::move(input_commands),
    std::move(commands)};
  input_commands.resize(0);
  commands.resize(0);
  return std::move(ret);
}

bool generate_commands_t::was_moved_to(tid_t tid, loc_t loc) const {
  if(moved_to_locs.count(tid) == 0) {
    return false;
  }

  for(auto const& other_loc: moved_to_locs.at(tid)) {
    if(other_loc == loc) {
      return true;
    }
  }
  return false;
}

void generate_commands_t::assure_moved_to(
  vector<command_ptr_t>& cmds,
  tid_t tid, loc_t from, loc_t to)
{
  if(was_moved_to(tid, to)) {
    return;
  }
  cmds.emplace_back(command_t::create_move(
    next_command_id(),
    {tid, from},
    {tid, to}));
  moved_to_locs[tid].push_back(to);
}

void generate_commands_t::add_deletes()
{
  // For each node that is not an output node or an input node,
  // and is not a no op, delete the tids
  for(nid_t nid = 0; nid != dag.size(); ++nid) {
    node_t const& node = dag[nid];
    if(node.ups.size() > 0 && node.downs.size() > 0 && !relations[nid].is_no_op()) {
      _add_deletes(nid);
    }
  }
}

void generate_commands_t::_add_deletes(nid_t nid)
{
  for(auto const& [tid, loc]: tid_locs[nid]) {
    commands.emplace_back(command_t::create_delete(
      next_command_id(),
      {tid_loc_t{tid, loc}}));
    for(auto const& other_loc: moved_to_locs[tid]) {
      commands.emplace_back(command_t::create_delete(
        next_command_id(),
        {tid_loc_t{tid, other_loc}}));
    }
  }
}

}}




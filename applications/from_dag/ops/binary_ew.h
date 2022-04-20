#include "types.h"

namespace _register_binary_ew {

// binary ew does this:
//   traversing the lhs, rhs arrays in order,
//     out[offset] = op(alpha*lhs[lhs_offset], beta*rhs[rhs_offset])
struct info_t {
  int which;
  float alpha; // scale the output

  vector<int64_t> dims;
  vector<int64_t> str_lhs;
  vector<int64_t> str_rhs;
};

// Now compute the strides
vector<int64_t> compute_stride(vector<int64_t> const& dims, vector<int> which)
{
  vector<int64_t> ret;
  ret.reserve(dims.size());

  int64_t p = 1;
  int idx_inn = 0;
  for(int idx_out = 0; idx_out != dims.size(); ++idx_out) {
    if(idx_out == which[idx_inn]) {
      ret.push_back(p);

      // Now we have to increase p
      p *= dims[idx_out];
      // And we can go on to the next index
      idx_inn++;

    } else {
      // this index does not come into play
      ret.push_back(0);
    }
  }

  assert(idx_inn == which.size());

  return ret;
};

void parse_dims_which(
  bbts::ud_impl_t::tensor_params_t const& params,
  cu_shape_t const& meta_lhs,
  cu_shape_t const& meta_rhs,
  vector<int64_t>& dims,
  vector<int>& which_lhs,
  vector<int>& which_rhs)
{
  which_lhs.resize(0);
  which_lhs.reserve(meta_lhs.rank);
  int i = 2;
  for(; i != 2 + meta_lhs.rank; ++i) {
    which_lhs.push_back(params.get_raw(i).i);
  }

  which_rhs.resize(0);
  which_rhs.reserve(meta_rhs.rank);
  int rest = i + meta_rhs.rank;
  for(; i != rest; ++i) {
    which_rhs.push_back(params.get_raw(i).i);
  }
  assert(i == params.num_parameters());

  int rank_out = 0;
  for(auto const& w: which_lhs) {
    rank_out = std::max(w + 1, rank_out);
  }
  for(auto const& w: which_rhs) {
    rank_out = std::max(w + 1, rank_out);
  }

  dims = vector<int64_t>(rank_out, 0);
  for(int idx_lhs = 0; idx_lhs != which_lhs.size(); ++idx_lhs) {
    int idx_out = which_lhs[idx_lhs];
    dims[idx_out] = meta_lhs.dims[idx_lhs];
  }

  for(int idx_rhs = 0; idx_rhs != which_rhs.size(); ++idx_rhs) {
    int idx_out = which_rhs[idx_rhs];
    if(dims[idx_out] > 0) {
      assert(dims[idx_out] == meta_rhs.dims[idx_rhs]);
    } else {
      dims[idx_out] = meta_rhs.dims[idx_rhs];
    }
  }

  for(auto const& d: dims) {
    assert(d > 0);
  }
}

info_t parse(
  bbts::ud_impl_t::tensor_params_t const& params,
  cu_shape_t const& meta_lhs,
  cu_shape_t const& meta_rhs)
{
  info_t ret;

  ret.which = params.get_int<0>();
  ret.alpha = params.get_float<1>();

  vector<int> lhs_which, rhs_which;
  parse_dims_which(
    params, meta_lhs, meta_rhs,
    ret.dims, lhs_which, rhs_which);

  ret.str_lhs = compute_stride(ret.dims, lhs_which);
  ret.str_rhs = compute_stride(ret.dims, rhs_which);

  return ret;
}

void set_out_meta(info_t info, cu_shape_t& meta_out)
{
  meta_out.rank = info.dims.size();
  for(int i = 0; i != info.dims.size(); ++i) {
    meta_out.dims[i] = info.dims[i];
  }
}

inline float _add(float lhs, float rhs){ return lhs + rhs; }
inline float _max(float lhs, float rhs){ return std::max(lhs, rhs); }
inline float _min(float lhs, float rhs){ return std::min(lhs, rhs); }
inline float _mul(float lhs, float rhs){ return lhs * rhs; }
inline float _sub(float lhs, float rhs){ return lhs - rhs; }
inline float _div(float lhs, float rhs){ return lhs / rhs; }

struct reference_indexer_t {
  reference_indexer_t(
    vector<int64_t> dims_,
    vector<int> lhs_which_,
    vector<int> rhs_which_)
    : idx(dims_.size(), 0), dims(dims_), lhs_which(lhs_which_), rhs_which(rhs_which_)
  {
    out_which.reserve(dims.size());
    for(int i = 0; i != dims.size(); ++i) {
      out_which.push_back(i);
    }
  }

  int64_t out() const { return get(out_which); }
  int64_t lhs() const { return get(lhs_which); }
  int64_t rhs() const { return get(rhs_which); }

  bool increment() {
    for(int i = 0; i < idx.size(); ++i) {
      if(idx[i] + 1 == dims[i]) {
        idx[i] = 0;
      } else {
        idx[i] += 1;
        return true;
      }
    }
    return false;
  }

//private:
  int64_t get(vector<int> which) const {
    int64_t ret = 0;
    int64_t p = 1;
    for(int const& w: which) {
      ret += p*idx[w];
      p *= dims[w];
    }
    return ret;
  }

  vector<int64_t> idx;

  vector<int64_t> const dims;
  vector<int>     const lhs_which;
  vector<int>     const rhs_which;
  vector<int>           out_which;
};

// Run the same computation
// to verify that the output in ous is correct.
void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  std::cout << "REFERENCE BINARY EW" << std::endl;
  std::string errmsg = "binary elementwise reference error. ";

  cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
  cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();

  float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
  float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
  float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

  int which = params.get_int<0>();
  float alpha = params.get_float<1>();

  vector<int64_t> dims;
  vector<int> lhs_which;
  vector<int> rhs_which;

  // assuming this parse is correct
  parse_dims_which(
    params, meta_lhs, meta_rhs,
    dims, lhs_which, rhs_which);

  decltype(_add) *binary_op;
  switch(which) {
    case 0: binary_op = _add; break;
    case 1: binary_op = _max; break;
    case 2: binary_op = _min; break;
    case 3: binary_op = _mul; break;
    case 4: binary_op = _sub; break;
    case 5: binary_op = _div; break;
  }

  reference_indexer_t indexer(dims, lhs_which, rhs_which);
  do {
    float v = alpha * binary_op(data_lhs[indexer.lhs()], data_rhs[indexer.rhs()]);

    float err = std::abs(v - data_out[indexer.out()]);

    if(err > 0.00001) {
      std::cout << indexer.idx << std::endl;
      std::cout << dims << ", " << lhs_which << ", " << rhs_which << std::endl;
      std::cout << v << ", " << err << std::endl;
      throw std::runtime_error(errmsg);
    };
  } while(indexer.increment());
}

struct op_t {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous)
  {
    cu_debug_write_t("binary_ew");

    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    set_out_meta(info, meta_out);

    float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
    float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_BINARY_EW_OFF

    decltype(_add) *binary_op;
    switch(info.which) {
      case 0: binary_op = _add; break;
      case 1: binary_op = _max; break;
      case 2: binary_op = _min; break;
      case 3: binary_op = _mul; break;
      case 4: binary_op = _sub; break;
      case 5: binary_op = _div; break;
    }

    auto const& dims = info.dims;
    auto const& sl   = info.str_lhs;
    auto const& sr   = info.str_rhs;

    if(dims.size() == 0) {
      data_out[0] = binary_op(data_lhs[0], data_rhs[0]);
    } else
    if(dims.size() == 1) {
      int64_t offset = 0;
      for(int64_t i0 = 0; i0 != dims[0]; ++i0) {
        data_out[offset++] = binary_op(
          data_lhs[i0*sl[0]],
          data_rhs[i0*sr[0]]);
      }
    } else
    if(dims.size() == 2) {
      int64_t offset = 0;
      for(int64_t i1 = 0; i1 != dims[1]; ++i1) {
      for(int64_t i0 = 0; i0 != dims[0]; ++i0) {
        data_out[offset++] = binary_op(
          data_lhs[i0*sl[0] + i1*sl[1]],
          data_rhs[i0*sr[0] + i1*sr[1]]);
      }}
    } else
    if(dims.size() == 3) {
      int64_t offset = 0;
      for(int64_t i2 = 0; i2 != dims[2]; ++i2) {
      for(int64_t i1 = 0; i1 != dims[1]; ++i1) {
      for(int64_t i0 = 0; i0 != dims[0]; ++i0) {
        data_out[offset++] = binary_op(
          data_lhs[i0*sl[0] + i1*sl[1] + i2*sl[1]],
          data_rhs[i0*sr[0] + i1*sr[1] + i2*sr[1]]);
      }}}
    } else
    if(dims.size() == 4) {
      int64_t offset = 0;
      for(int64_t i3 = 0; i3 != dims[3]; ++i3) {
      for(int64_t i2 = 0; i2 != dims[2]; ++i2) {
      for(int64_t i1 = 0; i1 != dims[1]; ++i1) {
      for(int64_t i0 = 0; i0 != dims[0]; ++i0) {
        data_out[offset++] = binary_op(
          data_lhs[i0*sl[0] + i1*sl[1] + i2*sl[1] + i3*sl[2]],
          data_rhs[i0*sr[0] + i1*sr[1] + i2*sr[1] + i3*sr[2]]);
      }}}}
    } else
    if(dims.size() == 5) {
      int64_t offset = 0;
      for(int64_t i4 = 0; i4 != dims[4]; ++i4) {
      for(int64_t i3 = 0; i3 != dims[3]; ++i3) {
      for(int64_t i2 = 0; i2 != dims[2]; ++i2) {
      for(int64_t i1 = 0; i1 != dims[1]; ++i1) {
      for(int64_t i0 = 0; i0 != dims[0]; ++i0) {
        data_out[offset++] = binary_op(
          data_lhs[i0*sl[0] + i1*sl[1] + i2*sl[1] + i3*sl[2] + i4*sl[3]],
          data_rhs[i0*sr[0] + i1*sr[1] + i2*sr[1] + i3*sr[2] + i4*sr[3]]);
      }}}}}
    } else {
      throw std::runtime_error("binary_ew only supports up to 5 dimensions");
    }

    if(info.alpha != 1.0) {
      int64_t n = product_dims(info.dims);
      cblas_sscal(n, info.alpha, data_out, 1);
    }

#ifdef CU_BARB_REFERENCE
    reference(params, ins, ous);
#endif
#endif
  }
};

struct f: public ud_impl_t {
  f(std::string name, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor", "cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = false;
    fn     = op;
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &ins) override
  {
    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    info_t info = parse(params, meta_lhs, meta_rhs);
    return product_dims(info.dims);
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &ins,
    meta_args_t &ous) const override
  {
    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    set_out_meta(info, meta_out);

    DCB01("lhs,rhs,out: " << meta_lhs << "," << meta_rhs << ", " << meta_out);
  }
};

}

void register_binary_ew(
  udf_manager_ptr udf_manager,
  std::string name)
{
  // make an f, do the thing.
  udf_manager->register_udf(std::make_unique<ud_func_t>(
    ud_func_t {
        .ud_name = name,
        .is_ass = false,
        .is_com = false,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
    }));

  udf_manager->register_udf_impl(
    std::make_unique<_register_binary_ew::f>(name, _register_binary_ew::op_t()));
}

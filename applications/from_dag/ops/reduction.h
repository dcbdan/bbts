#include "types.h"

namespace _register_reduction {

// This computation covers
//   ijb->jb

struct info_t {
  castable_op_t op;
  float alpha;

  vector<int64_t> b;
  vector<int64_t> i;
  vector<int64_t> j;

  vector<int64_t> dims() const {
    vector<int64_t> ret;
    ret.reserve(b.size() + j.size());

    for(auto& d: j) { ret.push_back(d); }
    for(auto& d: b) { ret.push_back(d); }

    return ret;
  }

  int64_t ni() const { return product_dims(i); }
  int64_t nj() const { return product_dims(j); }
  int64_t nb() const { return product_dims(b); }
};

info_t parse(
  bbts::ud_impl_t::tensor_params_t const& params,
  cu_shape_t const& meta_inn)
{
  // The params contains
  //   which op, alpha, ni, nj, nb

  info_t ret;

  ret.op = castable_op_t(params.get_int<0>());
  ret.alpha = params.get_int<1>();

  int ni = params.get_int<2>(); ret.i.reserve(ni);
  int nj = params.get_int<3>(); ret.j.reserve(nj);
  int nb = params.get_int<4>(); ret.b.reserve(nb);

  assert(ni + nj + nb == meta_inn.rank);

  for(int x = 0; x != ni; ++x) {
    ret.i.push_back(meta_inn.dims[x]);
  }
  for(int x = ni; x != ni + nj; ++x) {
    ret.j.push_back(meta_inn.dims[x]);
  }
  for(int x = ni + nj; x != ni + nj + nb; ++x) {
    ret.b.push_back(meta_inn.dims[x]);
  }

  return ret;
}

void set_out_meta(info_t info, cu_shape_t& meta_out)
{
  auto dims = info.dims();
  meta_out.rank = dims.size();
  for(int i = 0; i != dims.size(); ++i) {
    meta_out.dims[i] = dims[i];
  }
}

void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  std::cout << "REFERENCE REDUCTION" << std::endl;
  std::string errmsg = "reduction reference error. ";

  cu_shape_t const& meta_inn = ins.get<0>().as<cu_meta_t>().m();
  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();

  float* data_inn = (float*)(ins.get<0>().as<cu_t>().data());
  float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

  info_t info = parse(params, meta_inn);

  int di = info.i.size();
  int dj = info.j.size();
  int db = info.b.size();
  for(int x = 0; x != dj; ++x) {
    assert(meta_inn.dims[x + di] == meta_out.dims[x]);
  }
  for(int x = 0; x != db; ++x) {
    assert(meta_out.dims[x + di + dj] == meta_out.dims[x + dj]);
  }

  int64_t nb = info.nb();
  int64_t ni = info.ni();
  int64_t nj = info.nj();

  int64_t si_inn = 1;
  int64_t sj_inn = ni;
  int64_t sb_inn = ni*nj;

  int64_t sj_out = 1;
  int64_t sb_out = nj;

  for(int64_t b = 0; b != nb; ++b) {
  for(int64_t j = 0; j != nj; ++j) {
    float v = data_inn[0*si_inn + j*sj_inn + b*sb_inn];
    for(int64_t i = 1; i < ni; ++i) {
      v = info.op.scalar_op(v, data_inn[i*si_inn + j*sj_inn + b*sb_inn]);
    }
    v *= info.alpha;

    float err = std::abs(v - data_out[j*sj_out + b*sb_out]);
    if(err > 0.00001) {
      throw std::runtime_error(errmsg);
    };
  }}
}

struct op_t {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous)
  {
    cu_debug_write_t("reduction");

    cu_shape_t const& meta_inn = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_inn);
    set_out_meta(info, meta_out);

    float* data_inn = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_REDUCTION_OFF
    int64_t ni = info.ni();
    int64_t nj = info.nj();
    int64_t nb = info.nb();

    for(int64_t b = 0; b != nb; ++b) {
    for(int64_t j = 0; j != nj; ++j) {
      data_out[j + nj*b] = info.op.agg_op(data_inn + (ni*j + nj*b), ni);
    }}
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
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = false;
    fn     = op;
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &ins) override
  {
    cu_shape_t const& meta_inn = ins.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_inn);

    int64_t nb = info.nb();
    int64_t ni = info.ni();
    int64_t nj = info.nj();

    return nb*ni*nj;
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &ins,
    meta_args_t &ous) const override
  {
    cu_shape_t const& meta_inn = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_inn);
    set_out_meta(info, meta_out);
  }
};

}

void register_reduction(
  udf_manager_ptr udf_manager,
  std::string name)
{
  // make an f, do the thing.
  udf_manager->register_udf(std::make_unique<ud_func_t>(
    ud_func_t {
        .ud_name = name,
        .is_ass = false,
        .is_com = false,
        .num_in = 1,
        .num_out = 1,
        .impls = {}
    }));

  udf_manager->register_udf_impl(
    std::make_unique<_register_reduction::f>(name, _register_reduction::op_t()));
}

#include "types.h"

namespace _register_reduction {

// This computation covers
//   ij->j

struct info_t {
  castable_op_t op;
  float alpha;
  int64_t ni;
  int64_t nj;
};

// Params: which, alpha, ni, nj
info_t parse(
  bbts::ud_impl_t::tensor_params_t const& params)
{
  info_t ret;
  ret.op     = castable_op_t(params.get_int<0>());
  ret.alpha  = params.get_float<1>();
  ret.ni     = params.get_int<2>();
  ret.nj     = params.get_int<3>();
  return ret;
}

void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  throw std::runtime_error("not implemented");
//  std::cout << "REFERENCE REDUCTION" << std::endl;
//  std::string errmsg = "reduction reference error. ";
//
//  cu_shape_t const& meta_inn = ins.get<0>().as<cu_meta_t>().m();
//  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();
//
//  float* data_inn = (float*)(ins.get<0>().as<cu_t>().data());
//  float* data_out = (float*)(ous.get<0>().as<cu_t>().data());
//
//  info_t info = parse(params, meta_inn);
//
//  int di = info.i.size();
//  int dj = info.j.size();
//  for(int x = 0; x != dj; ++x) {
//    assert(meta_inn.dims[x + di] == meta_out.dims[x]);
//  }
//
//  int64_t ni = info.ni();
//  int64_t nj = info.nj();
//
//  for(int64_t j = 0; j != nj; ++j) {
//    float v = data_inn[j*ni];
//    for(int64_t i = 1; i < ni; ++i) {
//      v = info.op.scalar_op(v, data_inn[i + j*ni]);
//    }
//    v *= info.alpha;
//
//    float err = std::abs(v - data_out[j]);
//    if(err > 0.00001) {
//      throw std::runtime_error(errmsg);
//    };
//  }
}

struct op_t {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous)
  {
    cu_debug_write_t("reduction");

    info_t info = parse(params);

    int64_t& size_out = ous.get<0>().as<cu_meta_t>().size();
    size_out = info.nj;

    float* data_inn = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_REDUCTION_OFF
    int64_t const& ni = info.ni;
    int64_t const& nj = info.nj;

    for(int64_t j = 0; j != nj; ++j) {
      data_out[j] = info.op.agg_op(data_inn + ni*j, ni);
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
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = false;
    fn     = op;
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &ins) override
  {
    info_t info = parse(params);
    int64_t const& ni = info.ni;
    int64_t const& nj = info.nj;
    return ni*nj;
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &ins,
    meta_args_t &ous) const override
  {
    info_t info = parse(params);

    int64_t& size_out = ous.get<0>().as<cu_meta_t>().size();

    size_out = info.nj;
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

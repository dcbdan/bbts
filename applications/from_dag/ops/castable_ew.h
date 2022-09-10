#include "types.h"

namespace _register_castable_ew {

struct info_t {
  castable_op_t op;
  int64_t num_elem;
};

info_t parse(bbts::ud_impl_t::tensor_params_t const& params)
{
  info_t ret;
  ret.op = castable_op_t(params.get_int<0>());
  ret.num_elem = params.get_int<1>();
  return ret;
}

// Run the same computation
// to verify that the output in ous is correct.
void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  throw std::runtime_error("not implemented");
//  std::cout << "REFERENCE CASTABLE EW" << std::endl;
//  std::string errmsg = "castbale elementwise reference error. ";
//
//  cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
//  cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
//  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();
//
//  float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
//  float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
//  float* data_out = (float*)(ous.get<0>().as<cu_t>().data());
//
//  if(data_lhs == data_out || data_rhs == data_out) {
//    // This op is being done in place...
//    // TODO: what checks can be done here?
//    return;
//  }
//
//  castable_op_t cop(params.get_int<0>());
//
//  if(meta_out.rank != meta_lhs.rank) {
//    throw std::runtime_error(errmsg + " @ lhs rank");
//  }
//
//  if(meta_out.rank != meta_rhs.rank) {
//    throw std::runtime_error(errmsg + " @ rhs rank");
//  }
//
//  assert(meta_out.rank == meta_lhs.rank);
//  assert(meta_out.rank == meta_rhs.rank);
//
//  int64_t n = 1;
//  for(int i = 0; i != meta_out.rank; ++i) {
//    n *= meta_out.dims[i];
//    if(meta_out.dims[i] != meta_lhs.dims[i]) {
//      throw std::runtime_error(errmsg + " @ lhs dim");
//    }
//    if(meta_out.dims[i] != meta_rhs.dims[i]) {
//      throw std::runtime_error(errmsg + " @ rhs dim");
//    }
//  }
//
//  for(int64_t i = 0; i != n; ++i) {
//    float v = cop.scalar_op(data_lhs[i], data_rhs[i]);
//    float err = std::abs(data_out[i] - v);
//
//    if(err > 0.00001) {
//      std::cout << v << ", " << data_out[i] << std::endl;
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
    cu_debug_write_t("castable_ew");

    info_t info = parse(params);
    int64_t& size_out = ous.get<0>().as<cu_meta_t>().size();
    size_out = info.num_elem;

    float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
    float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_CASTABLE_EW_OFF
    info.op.mkl_op(info.num_elem, data_lhs, data_rhs, data_out);
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
    inputInplace = {0, 1};
    is_gpu = false;
    fn     = op;
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &ins) override
  {
    info_t info = parse(params);
    return info.num_elem;
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &ins,
    meta_args_t &ous) const override
  {
    int64_t& size_out = ous.get<0>().as<cu_meta_t>().size();

    info_t info = parse(params);
    size_out = info.num_elem;
  }
};

}

void register_castable_ew(
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
    std::make_unique<_register_castable_ew::f>(name, _register_castable_ew::op_t()));
}

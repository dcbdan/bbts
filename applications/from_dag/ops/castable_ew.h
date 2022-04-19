#include "types.h"

namespace _register_castable_ew {

struct info_t {
  vector<int64_t> dims;

  castable_op_t op;

  int64_t num_elem() const { return product_dims(dims); }
};

info_t parse(
  bbts::ud_impl_t::tensor_params_t const& params,
  cu_shape_t const& meta_lhs,
  cu_shape_t const& meta_rhs)
{
  info_t ret;
  ret.dims.reserve(meta_lhs.rank);
  for(int i = 0; i != meta_lhs.rank; ++i) {
    ret.dims.push_back(meta_lhs.dims[i]);
  }

  assert(params.num_parameters() == 1);

  ret.op = castable_op_t(params.get_int<0>());

  return ret;
}

void set_out_meta(info_t info, cu_shape_t& meta_out)
{
  meta_out.rank = info.dims.size();
  for(int i = 0; i != info.dims.size(); ++i) {
    meta_out.dims[i] = info.dims[i];
  }
}

// Run the same computation
// to verify that the output in ous is correct.
void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  std::cout << "REFERENCE CASTABLE EW" << std::endl;
  std::string errmsg = "castbale elementwise reference error. ";

  cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
  cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();

  float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
  float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
  float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

  if(data_lhs == data_out || data_rhs == data_out) {
    // This op is being done in place...
    // TODO: what checks can be done here?
    return;
  }

  castable_op_t cop(params.get_int<0>());

  if(meta_out.rank != meta_lhs.rank) {
    throw std::runtime_error(errmsg + " lhs rank");
  }

  if(meta_out.rank != meta_rhs.rank) {
    throw std::runtime_error(errmsg + " rhs rank");
  }

  int64_t n = 1;
  for(int i = 0; i != meta_out.rank; ++i) {
    n *= meta_out.dims[i];
    if(meta_out.dims[i] != meta_lhs.dims[i]) {
      throw std::runtime_error(errmsg + " lhs dim");
    }
    if(meta_out.dims[i] != meta_rhs.dims[i]) {
      throw std::runtime_error(errmsg + " rhs dim");
    }
  }

  for(int64_t i = 0; i != n; ++i) {
    float v = cop.scalar_op(data_lhs[i], data_rhs[i]);
    float err = std::abs(data_out[i] - v);

    if(err > 0.00001) {
      throw std::runtime_error(errmsg);
    };
  }
}

struct op_t {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous)
  {
    cu_debug_write_t("castable_ew");

    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    set_out_meta(info, meta_out);

    float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
    float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_CASTABLE_EW_OFF
    info.op.mkl_op(info.num_elem(), data_lhs, data_rhs, data_out);
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
    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    return product_dims(cu_shape_as_vec(meta_lhs));
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

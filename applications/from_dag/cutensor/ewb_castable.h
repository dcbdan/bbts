#include "cu.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>


namespace _register_ewb_castable {

using namespace _cutensor_utils;

using op_t = ud_impl_t::ud_impl_callable;

template <typename M>
void set_out_meta(
  M const& lhs,
  M& out) {
  out.rank = lhs.rank;
  for(int r = 0; r != lhs.rank; ++r) {
    out.dims[r] = lhs.dims[r];
  }
}

struct cpu_op {
  cpu_op() {}

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const
  {
    cu_debug_write_t("ewb_castable");

    castable_op_t op(params.get_int<0>());

    cu_shape_t const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(meta_lhs, meta_out);

    int n = _in.get<0>().as<cu_meta_t>().num_elem();

    float* data_lhs = (float*)(_in.get<0>().as<cu_t>().data());
    float* data_rhs = (float*)(_in.get<1>().as<cu_t>().data());
    float* data_out = (float*)(_out.get<0>().as<cu_t>().data());

#ifndef CU_EWB_CASTABLE_OFF
    op.mkl_op(n, data_lhs, data_rhs, data_out);
#endif
  }
};

struct f : public ud_impl_t {
  f(std::string name, bool is_gpu_, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor", "cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {0,1};
    is_gpu = is_gpu_;
    fn     = op;
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    return _in.get<0>().as<cu_meta_t>().num_elem();
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in,
    meta_args_t &_out) const override {

    auto const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(meta_lhs, meta_out);
  }

};

}

void register_ewb_castable(
  udf_manager_ptr udf_manager,
  std::string name)
{
  // make an f, do the thing.
  udf_manager->register_udf(std::make_unique<ud_func_t>(
    ud_func_t {
        .ud_name = name,
        .is_ass = true,
        .is_com = true,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
    }));
  udf_manager->register_udf_impl(
    std::make_unique<_register_ewb_castable::f>(
      name, false, _register_ewb_castable::cpu_op()));
}

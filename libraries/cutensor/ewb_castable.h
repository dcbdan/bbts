#include "cu.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>


namespace _register_ewb_castable {

using namespace _cutensor_utils;

template <typename M>
void set_out_meta(
  M const& lhs,
  M& out) {
  out.rank = lhs.rank;
  for(int r = 0; r != lhs.rank; ++r) {
    out.dims[r] = lhs.dims[r];
  }
}

struct op {
  op() : one(1.0) {
    std::iota(ordering, ordering + MAXRANK, 0);
  }

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const {

    castable_op_t op(params.get_int<0>());

    auto const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(meta_lhs, meta_out);

    void* data_lhs = _in.get<0>().as<cu_t>().data();
    void* data_rhs = _in.get<1>().as<cu_t>().data();
    void* data_out = _out.get<0>().as<cu_t>().data();

    cutensorTensorDescriptor_t desc;
    handle_error("cutensorInitTensorDescriptor same shape", cutensorInitTensorDescriptor(
        &params.cutensor_handle,
        &desc,
        meta_lhs.rank,
        meta_lhs.dims,
        NULL,
        cutensor_scalar_type,
        CUTENSOR_OP_IDENTITY));

    handle_error("cutensorElementwiseBinary", cutensorElementwiseBinary(
        &params.cutensor_handle,
        (void*)&one, data_lhs, &desc, ordering,
        (void*)&one, data_rhs, &desc, ordering,
                     data_out, &desc, ordering,
        op.cu_op, cutensor_scalar_type, params.stream));

  }

  float one;
  int ordering[MAXRANK];
};

struct f : public ud_impl_t {
  f(std::string name) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor", "cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {0,1};
    is_gpu = true;
    fn     = op();
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
  std::string name) {

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
    std::make_unique<_register_ewb_castable::f>(name));
}

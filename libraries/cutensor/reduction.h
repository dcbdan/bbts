#include "../cutensor/cu.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace _register_reduction {

using namespace _cutensor_utils;

struct params_t {
  castable_op_t bop;
  float alpha;
  modes_t ordering;
};

params_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  params_t ret;
  ret.bop   = castable_op_t(params.get_int<0>());
  ret.alpha = params.get_float<1>();
  for(int i = 2; i != params.num_parameters(); ++i) {
    ret.ordering.push_back(params.get_raw(i).i);
  }
  return ret;
}

template <typename M>
void set_out_meta(
  params_t const& p,
  M const& inn,
  M& out) {
  out.rank = p.ordering.size();
  for(int r = 0; r != out.rank; ++r) {
    out.dims[r] = inn.dims[p.ordering[r]];
  }
}

struct op {
  op() : zero(0.0) {
    std::iota(inn, inn + MAXRANK, 0);
  }

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const {

    params_t p = parse(params);

    auto const& meta_inn =  _in.get<0>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    if(maximum(p.ordering) >= meta_inn.rank) {
      throw std::invalid_argument("input rank is not great enough");
    }

    set_out_meta(p, meta_inn, meta_out);

    cutensorTensorDescriptor_t desc_inn;
    handle_error("red inn", cutensorInitTensorDescriptor(
      &params.cutensor_handle,
      &desc_inn,
      meta_inn.rank,
      meta_inn.dims,
      NULL, cutensor_scalar_type, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t desc_out;
    handle_error("red out", cutensorInitTensorDescriptor(
      &params.cutensor_handle,
      &desc_out,
      meta_out.rank,
      meta_out.dims,
      NULL, cutensor_scalar_type, CUTENSOR_OP_IDENTITY));

    void* data_inn = _in.get<0>().as<cu_t>().data();
    void* data_out = _out.get<0>().as<cu_t>().data();

    uint64_t worksize;

    handle_error("get workspace", cutensorReductionGetWorkspace(
      &params.cutensor_handle,
      data_inn, &desc_inn, inn,
      data_out, &desc_out, p.ordering.data(),
      data_out, &desc_out, p.ordering.data(),
      p.bop.cu_op, cutensor_compute_type,
      &worksize));

    void* work = nullptr;
    if(!misc_cuda_cuda_malloc(&work, worksize)) {
      work = nullptr;
      worksize = 0;
    }

    handle_error("reduction", cutensorReduction(
      &params.cutensor_handle,
      (void*)&(p.alpha),  data_inn, &desc_inn, inn,
      (void*)&zero, data_out, &desc_out, p.ordering.data(),
                    data_out, &desc_out, p.ordering.data(),
      p.bop.cu_op, cutensor_compute_type,
      work, worksize,
      params.stream));
  }

  int inn[MAXRANK];
  float zero;
};

struct f : public ud_impl_t {
  f(std::string name) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
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

    params_t p = parse(params);

    auto const& meta_inn =  _in.get<0>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    if(maximum(p.ordering) >= meta_inn.rank) {
      throw std::invalid_argument("input rank is not great enough");
    }

    set_out_meta(p, meta_inn, meta_out);
  }
};

}

void register_reduction(
  udf_manager_ptr udf_manager,
  std::string name) {

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
    std::make_unique<_register_reduction::f>(name));

}



#include "../cutensor/cu.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace _register_ewb {

using namespace _cutensor_utils;

struct params_t {
  int i;
  float alpha;
  float beta;
  modes_t ordering_lhs, ordering_rhs;
};

params_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  params_t ret;
  ret.i = params.get_int<0>();
  ret.alpha = params.get_float<1>();
  ret.beta  = params.get_float<2>();
  int rank_lhs = params.get_int<3>();
  int i = 4;
  for(; i != 4 + rank_lhs; ++i) {
    ret.ordering_lhs.push_back(params.get_raw(i).i);
  }
  int rank_rhs = params.get_raw(i++).i;
  for(; i != params.num_parameters(); ++i) {
    ret.ordering_rhs.push_back(params.get_raw(i).i);
  }
  return ret;
}

template <typename M>
void set_out_meta(
  params_t const& p,
  M const& lhs,
  M const& rhs,
  M& out) {
  out.rank = 1 + std::max(maximum(p.ordering_lhs), maximum(p.ordering_rhs));
  for(int r = 0; r != lhs.rank; ++r) {
    out.dims[p.ordering_lhs[r]] = lhs.dims[r];
  }
  for(int r = 0; r != rhs.rank; ++r) {
    out.dims[p.ordering_rhs[r]] = rhs.dims[r];
  }
}

struct op {
  op() {
    std::iota(ordering_out, ordering_out + MAXRANK, 0);
  }

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const {

    params_t p = parse(params);

    auto const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto const& meta_rhs =  _in.get<1>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(p, meta_lhs, meta_rhs, meta_out);

    void* data_lhs = _in.get<0>().as<cu_t>().data();
    void* data_rhs = _in.get<1>().as<cu_t>().data();
    void* data_out = _out.get<0>().as<cu_t>().data();

    auto create_desc = [&](
        int rank, const int64_t* dims,
        cutensorOperator_t op,
        cutensorTensorDescriptor_t& desc) {
      handle_error("cutensorInitTensorDescriptorLHS", cutensorInitTensorDescriptor(
          &params.cutensor_handle,
          &desc,
          rank,
          dims,
          NULL,
          cutensor_scalar_type,
          op));
    };

    /* The binary operators
        Add 0
        Max 1
        Min 2
        Mul 3
        Sub 4
        Div 5
    */
    cutensorTensorDescriptor_t desc_lhs, desc_rhs, desc_out;

    create_desc(meta_lhs.rank, meta_lhs.dims, CUTENSOR_OP_IDENTITY, desc_lhs);

    if(p.i == 5) {
      create_desc(meta_rhs.rank, meta_rhs.dims, CUTENSOR_OP_RCP, desc_rhs);
    } else {
      create_desc(meta_rhs.rank, meta_rhs.dims, CUTENSOR_OP_IDENTITY, desc_rhs);
    }

    create_desc(meta_out.rank, meta_out.dims, CUTENSOR_OP_IDENTITY, desc_out);

    float alpha = p.alpha;
    float beta  = p.beta;
    if(p.i == 4) {
      beta = -1.0*beta;
    }

    cutensorOperator_t op;
    if(p.i == 0 || p.i == 4) {
      op = CUTENSOR_OP_ADD;
    } else if(p.i == 1) {
      op = CUTENSOR_OP_MAX;
    } else if(p.i == 2) {
      op = CUTENSOR_OP_MIN;
    } else if(p.i == 3 || p.i == 5) {
      op = CUTENSOR_OP_MUL;
    } else  {
      throw std::invalid_argument("invalid binary op!");
    }

    handle_error("cutensorElementwiseBinary", cutensorElementwiseBinary(
        &params.cutensor_handle,
        (void*)&alpha, data_lhs, &desc_lhs, p.ordering_lhs.data(),
        (void*)&beta,  data_rhs, &desc_rhs, p.ordering_rhs.data(),
                       data_out, &desc_out, ordering_out,
        op, cutensor_scalar_type, params.stream));
  }
  int ordering_out[MAXRANK];
};

struct f : public ud_impl_t {
  f(std::string name) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor", "cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = true;
    fn     = op();
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    params_t p = parse(params);

    auto const& lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto const& rhs =  _in.get<1>().as<cu_meta_t>().m();

    int rank = 1 + std::max(maximum(p.ordering_lhs), maximum(p.ordering_rhs));
    dims_t dims(rank, 0);
    for(int r = 0; r != lhs.rank; ++r) {
      dims[p.ordering_lhs[r]] = lhs.dims[r];
    }
    for(int r = 0; r != rhs.rank; ++r) {
      dims[p.ordering_rhs[r]] = rhs.dims[r];
    }
    return product(dims);
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in,
    meta_args_t &_out) const override {

    params_t p = parse(params);

    auto const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto const& meta_rhs =  _in.get<1>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(p, meta_lhs, meta_rhs, meta_out);
  }

};

}

void register_ewb(
  udf_manager_ptr udf_manager,
  std::string name) {

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
    std::make_unique<_register_ewb::f>(name));
}


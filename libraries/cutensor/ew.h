#include "cu.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>


namespace _register_ew {

using namespace _cutensor_utils;

using op_t = ud_impl_t::ud_impl_callable;

struct params_t {
  unary_op_t uop;
  float alpha;
  modes_t ordering;
};

params_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  params_t ret;
  int i = 0;
  parse_uop(params, ret.uop, i); // parse_uop writes to ret.uop and updates i
  ret.alpha = params.get_raw(i++).f;
  for(; i != params.num_parameters(); ++i) {
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

struct cpu_op {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const
  {
    params_t p = parse(params);

    cu_shape_t const& meta_inn =  _in.get<0>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    if(maximum(p.ordering) >= meta_inn.rank) {
      throw std::invalid_argument("input rank is not great enough");
    }

    set_out_meta(p, meta_inn, meta_out);

    int n = _in.get<0>().as<cu_meta_t>().num_elem();

    float* data_inn = (float*)(_in.get<0>().as<cu_t>().data());
    float* data_out = (float*)(_out.get<0>().as<cu_t>().data());

    if(p.uop.i == 0) {
      // sigmoid
      // (I couldn't find a direct mkl implementation)
      vsExp(n, data_inn, data_out);
      for(int i = 0; i != n; ++i) {
        data_out[i] = data_out[i] / (1 + data_out[i]);
      }
    } else if(p.uop.i == 1 ){
      // exp
      vsExp(n, data_inn, data_out);
    } else if(p.uop.i == 2 ){
      // Square
      vsSqr(n, data_inn, data_out);
    } else if(p.uop.i == 3 ){
      // Relu
      float zero = 0.0;
      vsFmaxI(n, data_inn, 1, &zero, 0, data_out, 1);
    } else if(p.uop.i == 4 ){
      // Relu'
      // (it wasn't immediately obvious how to do this with mkl)
      for(int i = 0; i != n; ++i) {
        data_out[i] = data_inn[i] < 0.0 ? 0.0 : 1.0;
      }
    } else if(p.uop.i == 5 ){
      // sqrt
      vsSqrt(n, data_inn, data_out);
    } else if(p.uop.i == 6 ){
      // add a constant value
      vsAddI(n, data_inn, 1, &p.uop.f, 0, data_out, 1);
    }

    // now do an in place permutation (if needed)
    inplace_permute(cu_shape_as_vec(meta_inn), p.ordering, data_out);
  }

};

struct gpu_op {
  gpu_op() : one(1.0), two(2.0), zero(0.0) {
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

    void* data_inn = _in.get<0>().as<cu_t>().data();
    void* data_out = _out.get<0>().as<cu_t>().data();

    auto create_desc_inn = [&](cutensorOperator_t op, cutensorTensorDescriptor_t& desc) {
      handle_error("cutensorInitTensorDescriptorReluDeriv", cutensorInitTensorDescriptor(
          &params.cutensor_handle,
          &desc,
          meta_inn.rank,
          meta_inn.dims,
          NULL,
          cutensor_scalar_type,
          op));
    };

    auto create_desc_out = [&](cutensorOperator_t op, cutensorTensorDescriptor_t& desc) {
      handle_error("cutensorInitTensorDescriptorReluDeriv", cutensorInitTensorDescriptor(
          &params.cutensor_handle,
          &desc,
          meta_out.rank,
          meta_out.dims,
          NULL,
          cutensor_scalar_type,
          op));
    };

    auto ew_in_out = [&](
          float const& alpha,
          cutensorTensorDescriptor_t const& desc_inn,
          cutensorTensorDescriptor_t const& desc_out) {
      handle_error("cutensorPermutation_ew_in_out", cutensorPermutation(
        &params.cutensor_handle,
        (void*)&alpha,
        data_inn, &desc_inn, inn,
        data_out, &desc_out, p.ordering.data(),
        cutensor_scalar_type, params.stream));
    };

    auto ew_out_out = [&](
          float const& alpha,
          cutensorTensorDescriptor_t const& desc0,
          cutensorTensorDescriptor_t const& desc1) {
      handle_error("cutensorPermutation_ew_out_out", cutensorPermutation(
        &params.cutensor_handle,
        (void*)&alpha,
        data_out, &desc0, inn,
        data_out, &desc1, inn,
        cutensor_scalar_type, params.stream));
    };

    /* The unaray ops are
       0 Sigmoid                   <- 1 uop pass
       1 Exp                       <- 1 uop pass
       2 Square                    <- do binary Mul
       3 Relu                      <- 1 uop pass
       4 Reluderiv                 <- sigmoid .> (2*) .> floor
       5 Sqrt                      <- 1 uop pass
       6 AddScalar Float           <- do binary Addition
    */
    if(p.uop.i == 0) {
      cutensorTensorDescriptor_t desc_inn;
      create_desc_inn(CUTENSOR_OP_SIGMOID, desc_inn);

      cutensorTensorDescriptor_t desc_out;
      create_desc_out(CUTENSOR_OP_IDENTITY, desc_out);

      ew_in_out(p.alpha, desc_inn, desc_out);
    } else if(p.uop.i == 1) {
      cutensorTensorDescriptor_t desc_inn;
      create_desc_inn(CUTENSOR_OP_EXP, desc_inn);

      cutensorTensorDescriptor_t desc_out;
      create_desc_out(CUTENSOR_OP_IDENTITY, desc_out);

      ew_in_out(p.alpha, desc_inn, desc_out);
    } else if(p.uop.i == 2) {
      cutensorTensorDescriptor_t desc_inn;
      create_desc_inn(CUTENSOR_OP_IDENTITY, desc_inn);

      cutensorTensorDescriptor_t desc_out;
      create_desc_out(CUTENSOR_OP_IDENTITY, desc_out);

      bool can_do_binop = true;
      for(int r = 0; r != p.ordering.size(); ++r) {
        if(p.ordering[r] != r) {
          can_do_binop = false;
          break;
        }
      }
      // If there is no reordering of modes, we can do cutensorElementwiseBinary
      // directly (the last two desc have to be equal or cutensor will fail).
      // Otherwise, first transpose, then square.
      if(can_do_binop) {
        // alpha*x*x
        handle_error("cutensorElementwiseBinarySQUARE", cutensorElementwiseBinary(
          &params.cutensor_handle,
          (void*)&p.alpha, data_inn, &desc_inn, inn,
          (void*)&one,     data_inn, &desc_inn, inn,
                           data_out, &desc_inn, inn,
          CUTENSOR_OP_MUL, cutensor_scalar_type,
          params.stream));
      } else {
        handle_error("cutensorPermutationSQuarae", cutensorPermutation(
          &params.cutensor_handle,
          (void*)&one, data_inn, &desc_inn, inn,
                       data_out, &desc_out, p.ordering.data(),
          cutensor_scalar_type,
          params.stream));
        handle_error("cutensorElementwiseBinarySQUARE_", cutensorElementwiseBinary(
          &params.cutensor_handle,
          (void*)&p.alpha, data_out, &desc_out, inn,
          (void*)&one,     data_out, &desc_out, inn,
                           data_out, &desc_out, inn,
          CUTENSOR_OP_MUL, cutensor_scalar_type,
          params.stream));
      }
    } else if(p.uop.i == 3) {
      cutensorTensorDescriptor_t desc_inn;
      create_desc_inn(CUTENSOR_OP_RELU, desc_inn);

      cutensorTensorDescriptor_t desc_out;
      create_desc_out(CUTENSOR_OP_IDENTITY, desc_out);

      ew_in_out(p.alpha, desc_inn, desc_out);
    } else if(p.uop.i == 4) {
      cutensorTensorDescriptor_t desc_sigmoid;
      create_desc_inn(CUTENSOR_OP_SIGMOID, desc_sigmoid);

      cutensorTensorDescriptor_t desc_out;
      create_desc_out(CUTENSOR_OP_IDENTITY, desc_out);

      ew_in_out(two, desc_sigmoid, desc_out);

      cutensorTensorDescriptor_t desc_floor;
      create_desc_out(CUTENSOR_OP_FLOOR, desc_floor);

      ew_out_out(1.0, desc_floor, desc_out);

    } else if(p.uop.i == 5) {
      cutensorTensorDescriptor_t desc_inn;
      create_desc_inn(CUTENSOR_OP_SQRT, desc_inn);

      cutensorTensorDescriptor_t desc_out;
      create_desc_out(CUTENSOR_OP_IDENTITY, desc_out);

      ew_in_out(p.alpha, desc_inn, desc_out);
    } else if(p.uop.i == 6) {
      throw std::invalid_argument("TODO: implement add scalar");
    }
  }

  float one, two, zero;
  int inn[MAXRANK];
};

struct f : public ud_impl_t {
  f(std::string name, bool is_gpu_, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
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

void register_ew(
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
    std::make_unique<_register_ew::f>(name, true, _register_ew::gpu_op()));
  udf_manager->register_udf_impl(
    std::make_unique<_register_ew::f>(name, false, _register_ew::cpu_op()));
}



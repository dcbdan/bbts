#include "cu.h"

namespace _register_ewb_same_shape {
  template <typename M>
  void set_out_meta(
    const M& in, M& out) {

    out.rank = in.rank;
    for(int r = 0; r != in.rank; ++r) {
      out.dims[r] = in.dims[r];
    }
  }

  struct op {
    op(float alpha, float beta, cutensorOperator_t scalar_op) 
      : alpha(alpha), beta(beta), scalar_op(scalar_op) {

      for(int32_t i = 0; i != MAXRANK; ++i) {
        mode_names[i] = i;
      }
    }

    void operator()(
      const bbts::ud_impl_t::tensor_params_t &params,
      const tensor_args_t &_in, 
      tensor_args_t &_out) const {

      auto const& meta = _in.get<0>().as<cu_meta_t>().m();
      set_out_meta(meta, _out.get<0>().as<cu_meta_t>().m());
  
      void* data0 = _in.get<0>().as<cu_t>().data();
      void* data1 = _in.get<1>().as<cu_t>().data();
      void* data2 = _out.get<0>().as<cu_t>().data();
  
      cutensorTensorDescriptor_t desc;
      handle_error("cutensorInitTensorDescriptor", cutensorInitTensorDescriptor( 
          &params.cutensor_handle, 
          &desc, 
          meta.rank,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          CUTENSOR_OP_IDENTITY));
 
      handle_error("cutensorElementwiseBinary", cutensorElementwiseBinary(
        // handle
        &params.cutensor_handle, 
        // input 0
        (void*)&alpha, data0, &desc, mode_names,
        // input 1
        (void*)&beta,  data1, &desc, mode_names,
        // output
                       data2, &desc, mode_names,
        // no output op; which scalar type
        scalar_op, cutensor_scalar_type,
        // the stream
        params.stream));
    }

    int32_t mode_names[MAXRANK];
    float alpha;
    float beta;
    cutensorOperator_t scalar_op;
  };

  struct f: public ud_impl_t {
    f(std::string name, float alpha, float beta, cutensorOperator_t scalar_op) {
      impl_name = name;
      ud_name = name;
      inputTypes = {"cutensor", "cutensor"};
      outputTypes = {"cutensor"};
      inputInplace = {0,1};
      is_gpu = true;
      fn     = op(alpha, beta, scalar_op);
    } 

    // returns an estimate of the complexity
    size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                               const meta_args_t &_in) override {
      auto const& in0 = _in.get<0>().as<cu_meta_t>().m();
      size_t ret = 1;
      for(int r = 0; r != in0.rank; ++r) {
        ret *= in0.dims[r];
      }
      return ret;
    }

    void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                      const meta_args_t &_in, meta_args_t &_out) const override {
      set_out_meta(
        _in.get<0>( ).as<cu_meta_t>().m(),
        _out.get<0>().as<cu_meta_t>().m());
    }
  };
}

void register_ewb_same_shape(
  udf_manager_ptr udf_manager,
  std::string name,
  cutensorOperator_t scalar_op,
  float alpha,
  float beta,
  bool is_ass,
  bool is_com) {

  // make an f, do the thing. 
  udf_manager->register_udf(std::make_unique<ud_func_t>(
    ud_func_t {
        .ud_name = name,
        .is_ass = is_ass,
        .is_com = is_com,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
    }));
  udf_manager->register_udf_impl(
    std::make_unique<_register_ewb_same_shape::f>(name, alpha, beta, scalar_op));
}

namespace _register_ew_same_shape {
  template <typename M>
  void set_out_meta(
    const M& in, M& out) {

    out.rank = in.rank;
    for(int r = 0; r != in.rank; ++r) {
      out.dims[r] = in.dims[r];
    }
  }

  struct op {
    op(float alpha, cutensorOperator_t scalar_op) 
      : alpha(alpha), scalar_op(scalar_op) {
      for(int32_t i = 0; i != MAXRANK; ++i) {
        mode_names[i] = i;
      }
    }

    void operator()(
      const bbts::ud_impl_t::tensor_params_t &params,
      const tensor_args_t &_in, 
      tensor_args_t &_out) const {

      auto const& meta = _in.get<0>().as<cu_meta_t>().m();
      set_out_meta(meta, _out.get<0>().as<cu_meta_t>().m());
  
      void* data0 = _in.get<0>( ).as<cu_t>().data();
      void* data1 = _out.get<0>().as<cu_t>().data();
  
      cutensorTensorDescriptor_t desc0_apply_op;
      handle_error("cutensorInitTensorDescriptor", cutensorInitTensorDescriptor( 
          &params.cutensor_handle, 
          &desc0_apply_op, 
          meta.rank,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          scalar_op));
      cutensorTensorDescriptor_t desc1;
      handle_error("cutensorInitTensorDescriptor", cutensorInitTensorDescriptor( 
          &params.cutensor_handle, 
          &desc1, 
          meta.rank,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          CUTENSOR_OP_IDENTITY));
 
      handle_error("cutensorElementwiseBinary", cutensorPermutation(
        // handle
        &params.cutensor_handle, 
        // input 0
        (void*)&alpha, data0, &desc0_apply_op, mode_names,
        // output
                       data1, &desc1,          mode_names,
        // which scalar type
        cutensor_scalar_type,
        // the stream
        params.stream));
    }

    int32_t mode_names[MAXRANK];
    float alpha;
    cutensorOperator_t scalar_op;
  };

  struct f: public ud_impl_t {
    f(std::string name, float alpha, cutensorOperator_t scalar_op) {
      impl_name = name;
      ud_name = name;
      inputTypes = {"cutensor"};
      outputTypes = {"cutensor"};
      inputInplace = {0};
      is_gpu = true;
      fn     = op(alpha, scalar_op);
    } 

    // returns an estimate of the complexity
    size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                               const meta_args_t &_in) override {
      auto const& in0 = _in.get<0>().as<cu_meta_t>().m();
      size_t ret = 1;
      for(int r = 0; r != in0.rank; ++r) {
        ret *= in0.dims[r];
      }
      return ret;
    }

    void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                      const meta_args_t &_in, meta_args_t &_out) const override {
      set_out_meta(
        _in.get<0>( ).as<cu_meta_t>().m(),
        _out.get<0>().as<cu_meta_t>().m());
    }
  };
}

void register_ew_same_shape(
  udf_manager_ptr udf_manager,
  std::string name,
  float alpha,
  cutensorOperator_t scalar_op) {

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
    std::make_unique<_register_ew_same_shape::f>(name, alpha, scalar_op));
}



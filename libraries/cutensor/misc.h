#include "cu.h"

namespace misc {
  template <typename M>
  void copy_meta(
    const M& in, M& out) {

    out.rank = in.rank;
    for(int r = 0; r != in.rank; ++r) {
      out.dims[r] = in.dims[r];
    }
  }
}

namespace _register_reluderiv {
  template <typename M>
  void set_out_meta(
    const M& in, M& out) {

    out.rank = in.rank;
    for(int r = 0; r != in.rank; ++r) {
      out.dims[r] = in.dims[r];
    }
  }

  struct op {
    op() 
      : two(2.0), one(1.0) {
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
      void* data1 = _out.get<0>().as<cu_t>().data();

      // reluderiv == sigmoid .> (2*) .> floor 
      //   first take the sigmoid, multiply by 2, then take the floor.
      cutensorTensorDescriptor_t desc_sigmoid;
      handle_error("cutensorInitTensorDescriptorReluDeriv", cutensorInitTensorDescriptor( 
          &params.cutensor_handle, 
          &desc_sigmoid, 
          meta.rank,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          CUTENSOR_OP_SIGMOID));
      cutensorTensorDescriptor_t desc_floor;
      handle_error("cutensorInitTensorDescriptorReluDeriv", cutensorInitTensorDescriptor( 
          &params.cutensor_handle, 
          &desc_floor, 
          meta.rank,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          CUTENSOR_OP_FLOOR));
      cutensorTensorDescriptor_t desc;
      handle_error("cutensorInitTensorDescriptorReluDeriv", cutensorInitTensorDescriptor( 
          &params.cutensor_handle, 
          &desc, 
          meta.rank,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          CUTENSOR_OP_IDENTITY));

      handle_error("cutensorPermutationReluDeriv0", cutensorPermutation(
        // handle
        &params.cutensor_handle, 
        // input 0
        (void*)&two, data0, &desc_sigmoid, mode_names,
        // output
                     data1, &desc,         mode_names,
        // which scalar type
        cutensor_scalar_type,
        // the stream
        params.stream));


      handle_error("cutensorPermutationReluDeriv1", cutensorPermutation(
        // handle
        &params.cutensor_handle, 
        // input 0
        (void*)&one, data1, &desc_floor, mode_names,
        // output
                     data1, &desc,      mode_names,
        // which scalar type
        cutensor_scalar_type,
        // the stream
        params.stream));
    }

    int32_t mode_names[MAXRANK];
    float two;
    float one;
  };

  struct f: public ud_impl_t {
    f(std::string name) {
      impl_name = name;
      ud_name = name;
      inputTypes = {"cutensor"};
      outputTypes = {"cutensor"};
      inputInplace = {0};
      is_gpu = true;
      fn     = op();
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

void register_reluderiv(
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
    std::make_unique<_register_reluderiv::f>(name));
}

namespace _register_square {
  template <typename M>
  void set_out_meta(
    const M& in, M& out) {

    out.rank = in.rank;
    for(int r = 0; r != in.rank; ++r) {
      out.dims[r] = in.dims[r];
    }
  }

  struct op {
    op() 
      : one(1.0) {

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
      void* data1 = _out.get<0>().as<cu_t>().data();
  
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
        (void*)&one, data0, &desc, mode_names,
        // input 1
        (void*)&one, data0, &desc, mode_names,
        // output
                     data1, &desc, mode_names,
        // no output op; which scalar type
        CUTENSOR_OP_MUL, cutensor_scalar_type,
        // the stream
        params.stream));
    }

    int32_t mode_names[MAXRANK];
    float one;
  };

  struct f: public ud_impl_t {
    f(std::string name) {
      impl_name = name;
      ud_name = name;
      inputTypes = {"cutensor"};
      outputTypes = {"cutensor"};
      inputInplace = {0};
      is_gpu = true;
      fn     = op();
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

void register_square(
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
    std::make_unique<_register_square::f>(name));
}

namespace _register_div_ij_i {
  template <typename M>
  void set_out_meta(
    const M& in, M& out) {

    out.rank = in.rank;
    for(int r = 0; r != in.rank; ++r) {
      out.dims[r] = in.dims[r];
    }
  }

  struct op {
    op() 
      : one(1.0) {

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
          2,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t desc1_recip;
      handle_error("cutensorInitTensorDescriptor", cutensorInitTensorDescriptor( 
          &params.cutensor_handle, 
          &desc1_recip, 
          1,
          meta.dims,
          NULL,
          cutensor_scalar_type,
          CUTENSOR_OP_RCP));
 
      // (1/B(i)) * (A(ij)) -> ij
      handle_error("cutensorElementwiseBinaryIJI", cutensorElementwiseBinary(
        // handle
        &params.cutensor_handle, 
        // input 1
        (void*)&one, data1, &desc1_recip, mode_names,
        // input 0
        (void*)&one, data0, &desc,        mode_names,
        // output
                     data2, &desc,        mode_names,
        // no output op; which scalar type
        CUTENSOR_OP_MUL, cutensor_scalar_type,
        // the stream
        params.stream));
    }

    int32_t mode_names[MAXRANK];
    float one;
  };

  struct f: public ud_impl_t {
    f(std::string name) {
      impl_name = name;
      ud_name = name;
      inputTypes = {"cutensor", "cutensor"};
      outputTypes = {"cutensor"};
      inputInplace = {0};
      is_gpu = true;
      fn     = op();
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


void register_div_ij_i(
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
    std::make_unique<_register_div_ij_i::f>(name));
}

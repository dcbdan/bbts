#include "cu.h"

void handle_error(std::string msg, cutensorStatus_t status) {
  if(status == CUTENSOR_STATUS_SUCCESS){
    return;
  }

  // TODO: violently die
  std::cout << "NOT HANDLING ERROR{" << msg << "}" << std::endl;
  switch(status) {
    case CUTENSOR_STATUS_NOT_INITIALIZED:        std::cout << "NOT_INITIALIZED"       ; break;
    case CUTENSOR_STATUS_ALLOC_FAILED:           std::cout << "ALLOC_FAILED"          ; break;
    case CUTENSOR_STATUS_INVALID_VALUE:          std::cout << "INVALID_VALUE"         ; break;
    case CUTENSOR_STATUS_ARCH_MISMATCH:          std::cout << "ARCH_MISMATCH"         ; break;
    case CUTENSOR_STATUS_MAPPING_ERROR:          std::cout << "MAPPING_ERROR"         ; break;
    case CUTENSOR_STATUS_EXECUTION_FAILED:       std::cout << "EXECUTION_FAILED"      ; break;
    case CUTENSOR_STATUS_INTERNAL_ERROR:         std::cout << "INTERNAL_ERROR"        ; break;
    case CUTENSOR_STATUS_NOT_SUPPORTED:          std::cout << "NOT_SUPPORTED"         ; break;
    case CUTENSOR_STATUS_LICENSE_ERROR:          std::cout << "LICENSE_ERROR"         ; break;
    case CUTENSOR_STATUS_CUBLAS_ERROR:           std::cout << "CUBLAS_ERROR"          ; break;
    case CUTENSOR_STATUS_CUDA_ERROR:             std::cout << "CUDA_ERROR"            ; break;
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE: std::cout << "INSUFFICIENT_WORKSPACE"; break;
    case CUTENSOR_STATUS_INSUFFICIENT_DRIVER:    std::cout << "INSUFFICIENT_DRIVER"   ; break;
    case CUTENSOR_STATUS_IO_ERROR:               std::cout << "IO_ERROR"              ; break;
  }
  std::cout << std::endl;
}

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

      std::cout << "OPERATOR ! " << std::endl;

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
      inputInplace = {}; //{0,1};
      is_gpu = true;
      fn     = op(alpha, beta, scalar_op);
      //gpu_fn = op(alpha, beta, scalar_op); 
    } 

    // returns an estimate of the complexity
    size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                               const meta_args_t &_in) override {
      auto const& in0 = _in.get<0>().as<cu_meta_t>().m();
      size_t ret = 0;
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
  cutensorOperator_t scalar_op_,
  float alpha_,
  float beta_,
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
    std::make_unique<_register_ewb_same_shape::f>(name, alpha_, beta_, scalar_op_));
}

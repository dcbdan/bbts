#include "cu.h"

#include <random>

using namespace bbts;

namespace _register_init {

  template <typename M>
  void set_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    M& out) {

    out.rank = params.num_parameters();
    for(int r = 0; r != out.rank; ++r){
      out.dims[r] = params._params[r].i; 
    }

  }

  struct op {
    op(float value) : value(value) {}

    void operator()(
      const bbts::ud_impl_t::tensor_params_t &params,
      const tensor_args_t &_in, 
      tensor_args_t &_out) const {

      set_out_meta(params, _out.get<0>().as<cu_meta_t>().m());

      cu_t& out = _out.get<0>().as<cu_t>();
      float* data = (float*)out.data();
      int n = out.meta().get_data_size() / sizeof(float);
      for(int idx = 0; idx != n; ++idx) {
        data[idx] = value;
      }
    }

    float value;
  };

  struct f : public ud_impl_t {
    f(std::string name, float value) {
      impl_name = name;
      ud_name = name;
      inputTypes = {};
      outputTypes = {"cutensor"};
      inputInplace = {};
      is_gpu = false;
      fn = op(value);
    }
  
    // returns an estimate of the complexity
    size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                               const meta_args_t &_in) override {
      return 0;
    }
  
    // return the meta of the output
    void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                      const meta_args_t &_in, meta_args_t &_out) const override {
      set_out_meta(params, _out.get<0>().as<cu_meta_t>().m());
    }
  };
}

void register_init(
  udf_manager_ptr udf_manager,
  std::string name,
  float value_) {

  udf_manager->register_udf(std::make_unique<ud_func_t>(
        ud_func_t {
          .ud_name = name,
          .is_ass = false,
          .is_com = false,
          .num_in = 0,
          .num_out = 1,
          .impls = {}
        }));
  udf_manager->register_udf_impl(std::make_unique<_register_init::f>(name, value_));

}

namespace _register_random {

  template <typename M>
  void set_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    M& out) {

    out.rank = params.num_parameters();
    for(int r = 0; r != out.rank; ++r){
      out.dims[r] = params._params[r].i; 
    }

  }

  struct op {
    op(float lo, float hi) : lo(lo), hi(hi) {}

    void operator()(
      const bbts::ud_impl_t::tensor_params_t &params,
      const tensor_args_t &_in, 
      tensor_args_t &_out) const {

      set_out_meta(params, _out.get<0>().as<cu_meta_t>().m());

      cu_t& out = _out.get<0>().as<cu_t>();
      float* data = (float*)out.data();
      int n = out.meta().get_data_size() / sizeof(float);

      std::random_device rd;  
      std::mt19937 gen(rd()); 
      std::uniform_real_distribution<> dis(lo, hi);

      for(int idx = 0; idx != n; ++idx) {
        data[idx] = dis(gen);
      }
    }

    float lo, hi;
  };

  struct f : public ud_impl_t {
    f(std::string name, float lo, float hi) {
      impl_name = name;
      ud_name = name;
      inputTypes = {};
      outputTypes = {"cutensor"};
      inputInplace = {};
      is_gpu = false;
      fn = op(lo, hi);
    }
  
    // returns an estimate of the complexity
    size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                               const meta_args_t &_in) override {
      return 0;
    }
  
    // return the meta of the output
    void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                      const meta_args_t &_in, meta_args_t &_out) const override {
      set_out_meta(params, _out.get<0>().as<cu_meta_t>().m());
    }
  };

}

void register_init_random(
  udf_manager_ptr udf_manager,
  std::string name,
  float lo,
  float hi) {

  udf_manager->register_udf(std::make_unique<ud_func_t>(
        ud_func_t {
          .ud_name = name,
          .is_ass = false,
          .is_com = false,
          .num_in = 0,
          .num_out = 1,
          .impls = {}
        }));
  udf_manager->register_udf_impl(std::make_unique<_register_random::f>(name, lo, hi));
}



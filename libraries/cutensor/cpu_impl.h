#include "cu.h"

namespace _cpu_impl {
  
  template <typename M>
  void set_out_meta_add(
    M const& x,
    M& z) {
    z = x;
  }

  struct add : public ud_impl_t {
    add(std::string name, std::string my_name) {
      impl_name = name;
      ud_name = my_name;
      inputTypes = {"cutensor", "cutensor"};
      outputTypes = {"cutensor"};
      inputInplace = {};
      is_gpu = false;
      fn = &add::fn_;
    }
  
    // returns an estimate of the complexity
    size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                               const meta_args_t &_in) override {
      auto const& meta_in = _in.get<0>().as<cu_meta_t>();
      return meta_in.num_elem();
    }
  
    // return the meta of the output
    void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                      const meta_args_t &_in, meta_args_t &_out) const override {
      set_out_meta_add(
        _in.get<0>( ).as<cu_meta_t>().m(),
        _out.get<0>().as<cu_meta_t>().m());
    }

    static void fn_(
      const bbts::ud_impl_t::tensor_params_t &params,
      const tensor_args_t &ins, 
      tensor_args_t &ous) {

      auto const& meta_in0 = ins.get<0>().as<cu_meta_t>();
      set_out_meta_add(meta_in0.m(), ous.get<0>().as<cu_meta_t>().m());

      float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
      float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
      float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

      size_t n = meta_in0.num_elem();
      for(int i = 0; i != n; ++i) {
        data_out[i] + data_lhs[i] + data_rhs[i];
      }
    }
  };
}

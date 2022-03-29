#pragma once

#include <cassert>
#include <functional>
#include "../ud_function.h"

namespace bbts {

struct dense_index_t : public ud_impl_t {

  // initializes the function
  dense_index_t(std::function<float(int,int)> by_index);

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in,
                    meta_args_t &_out) const override;

  // does the work
  static void f(std::function<float(int,int)> by_index,
                const bbts::ud_impl_t::tensor_params_t &params,
                const tensor_args_t &_in,
                tensor_args_t &_out);

  static ud_func_ptr_t get_ud_func() {
    return std::make_unique<ud_func_t>(
      ud_func_t {
          .ud_name = "index",
          .is_ass =false,
          .is_com = false,
          .num_in = 0,
          .num_out = 1,
          .impls = {}
      });
  }

};

}

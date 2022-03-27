#pragma once

#include <cassert>
#include "../ud_function.h"

namespace bbts {

struct dense_expand_t : public ud_impl_t {

  // initializes the function
  dense_expand_t();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_inn) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_inn, meta_args_t &_out) const override;

  // does the work
  static void f(const bbts::ud_impl_t::tensor_params_t &params,
                const tensor_args_t &_inn,
                tensor_args_t &_out);

  // determine whether or not doing a compact on this input is of any utility
  bool can_compact(
    uint32_t which_input,
    const bbts::ud_impl_t::tensor_params_t &params_without_extra_info,
    const meta_args_t &_inn) const;
};

}

#include "dense_iota.h"
#include "../../tensor/builtin_formats.h"
#include <numeric>

bbts::dense_iota_t::dense_iota_t() {
  impl_name = "dense_iota";
  ud_name = "iota";

  inputTypes = {};
  outputTypes = {"dense"};

  inputInplace = {};

  is_gpu = false;

  fn = &dense_iota_t::f;
}

size_t bbts::dense_iota_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                               const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() < 2){
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void bbts::dense_iota_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the new values
  m_out = { params.get_uint<0>(),  params.get_uint<1>() };
}

void bbts::dense_iota_t::f(
  const bbts::ud_impl_t::tensor_params_t &params,
  const bbts::ud_impl_t::tensor_args_t &_in,
  bbts::ud_impl_t::tensor_args_t &_out)
{
  auto &out = _out.get<0>().as<dense_tensor_t>();
  auto &m_out = out.meta().m();

  m_out = {
    .num_rows = (uint32_t) params.get_uint<0>(),
    .num_cols = (uint32_t) params.get_uint<1>()
  };
  auto num_elem = m_out.num_rows * m_out.num_cols;

  float init_val = params.get_float<2>();

  std::iota(out.data(), out.data() + num_elem, init_val);
}

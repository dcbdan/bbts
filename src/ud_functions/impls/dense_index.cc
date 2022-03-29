#include "dense_index.h"
#include "../../tensor/builtin_formats.h"
#include <numeric>

bbts::dense_index_t::dense_index_t(
  std::function<float(int,int)> by_index)
{
  impl_name = "dense_index";
  ud_name = "index";

  inputTypes = {};
  outputTypes = {"dense"};

  inputInplace = {};

  is_gpu = false;

  using namespace std::placeholders;
  fn = std::bind(&bbts::dense_index_t::f, by_index, _1, _2, _3);
}

size_t bbts::dense_index_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                               const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() != 4){
    throw std::runtime_error("Incorrect number of parameters");
  }

  uint32_t bi = params.get_uint<0>();
  uint32_t ei = params.get_uint<1>();

  uint32_t bj = params.get_uint<2>();
  uint32_t ej = params.get_uint<3>();

  // O(n * m)
  return (ei-bi)*(ej-bj);
}

void bbts::dense_index_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {
  uint32_t bi = params.get_uint<0>();
  uint32_t ei = params.get_uint<1>();

  uint32_t bj = params.get_uint<2>();
  uint32_t ej = params.get_uint<3>();

  // set the new values
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();
  m_out = { ei-bi, ej-bj  };
}

void bbts::dense_index_t::f(
  std::function<float(int,int)> by_index,
  const bbts::ud_impl_t::tensor_params_t &params,
  const bbts::ud_impl_t::tensor_args_t &_in,
  bbts::ud_impl_t::tensor_args_t &_out)
{
  auto &out = _out.get<0>().as<dense_tensor_t>();
  auto &m_out = out.meta().m();
  float* data = out.data();

  uint32_t bi = params.get_uint<0>();
  uint32_t ei = params.get_uint<1>();

  uint32_t bj = params.get_uint<2>();
  uint32_t ej = params.get_uint<3>();

  uint32_t ni = ei - bi;
  uint32_t nj = ej - bj;

  m_out = {
    .num_rows = ni,
    .num_cols = nj
  };

  for(int i = bi; i != ei; ++i) {
  for(int j = bj; j != ej; ++j) {
    data[nj*(i-bi) + (j-bj)] = by_index(i,j);
  }}
}

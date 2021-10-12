#include "dense_uniform.h"
#include "../../tensor/builtin_formats.h"
#include <cstdint>
#include <mkl_cblas.h>
#include <mkl.h>

bbts::dense_uniform_t::dense_uniform_t() {

  // set the names
  impl_name = "dense_uniform";
  ud_name = "uniform";

  // set the input and output types
  inputTypes = {};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &dense_uniform_t::uniform_rand;
}

size_t bbts::dense_uniform_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() < 2){
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void bbts::dense_uniform_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the new values
  m_out = { params.get_uint<0>(),  params.get_uint<1>() };
}

void bbts::dense_uniform_t::uniform_rand(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {


  // make the random stream
  VSLStreamStatePtr stream;
  vslNewStream(&stream, VSL_BRNG_MCG31, time(nullptr));

  // get the dense tensor
  auto &out = _out.get<0>().as<dense_tensor_t>();
  auto &m_out = out.meta().m();

  // the number of rows and columns
  auto numRows = (uint32_t) params.get_int<0>();
  auto numCols = (uint32_t) params.get_int<1>();

  // the left and right boundary
  auto left = params.get_float_or_default<2>(0.0f);
  auto right = params.get_float_or_default<3>(1.0f);

  // set the new meta data
  m_out = {.num_rows = numRows, .num_cols = numCols};

  // create a bunch of random numbers
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int32_t) (numRows * numCols), out.data(), left, right);

  // delete the stream
  vslDeleteStream(&stream);
}

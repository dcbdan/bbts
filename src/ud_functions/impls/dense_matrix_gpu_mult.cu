#include "dense_matrix_gpu_mult.h"
#include "../../tensor/builtin_formats.h"
#include <mkl_cblas.h>
#include <mkl.h>
#include <cublas_v2.h>

bbts::dense_matrix_gpu_mult_t::dense_matrix_gpu_mult_t() {

  // set the names
  impl_name = "dense_matrix_gpu_mult";
  ud_name = "matrix_mult";

  // set the input and output types
  inputTypes = {"dense", "dense"};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 0};

  // this is a CPU dense add
  is_gpu = true;

  // set the function that actually performs the add
  fn = &dense_matrix_gpu_mult_t::mult;
}

size_t bbts::dense_matrix_gpu_mult_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m * k)
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  const auto &m_b = _in.get<1>().as<dense_tensor_meta_t>().m();
  return m_a.num_rows * m_a.num_cols * m_b.num_cols;
}

void bbts::dense_matrix_gpu_mult_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                             const bbts::ud_impl_t::meta_args_t &_in,
                                             bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  const auto &m_b = _in.get<1>().as<dense_tensor_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the output
  m_out = {m_a.num_rows, m_b.num_cols};
}

void bbts::dense_matrix_gpu_mult_t::mult(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<dense_tensor_t>();
  auto &b = _in.get<1>().as<dense_tensor_t>();
  auto &out = _out.get<0>().as<dense_tensor_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // get the sizes
  uint32_t I = m_a.num_rows;
  uint32_t J = m_b.num_cols;
  uint32_t K = m_a.num_cols;

  // make sure the matrix size matches, this is only present during the debug build
  assert(m_a.num_cols == m_b.num_rows);

  // get the ptrs
  float *outData = out.data();
  float *in1Data = a.data();
  float *in2Data = b.data();

  // run the matrix multiply
  float alpha=1.0f;
  float beta=0.0f;
  cublasSgemm(params.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, I, J, K, &alpha, in1Data, K, in2Data, J, &beta, outData, J);

  // set the new meta data
  m_out = {m_a.num_rows, m_b.num_cols};
}

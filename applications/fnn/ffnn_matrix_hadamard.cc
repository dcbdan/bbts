#include "ffnn_matrix_hadamard.h"
#include "ffnn_types.h"

bbts::ffnn_matrix_hadamard::ffnn_matrix_hadamard() {

  // set the names
  impl_name = "ffnn_matrix_hadamard_cpu";
  ud_name = "ffnn_matrix_hadamard";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU kernel
  is_gpu = false;

  // set the function that actually performs the product
  fn = &ffnn_matrix_hadamard::mult;
}

size_t bbts::ffnn_matrix_hadamard::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_matrix_hadamard::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the new values
  m_out = { m_a.num_rows, m_a.num_cols, m_a.has_bias };
}

void bbts::ffnn_matrix_hadamard::mult(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &b = _in.get<1>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // make sure the matrix size matches, this is only
  // present during the debug build
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);
  assert(m_a.has_bias == m_b.has_bias);

  // set the new meta data
  m_out = {m_a.num_rows, m_a.num_cols, m_a.has_bias};

  // multiply a and b
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out.data()[row * m_a.num_cols + col] = a.data()[row * m_a.num_cols + col] *
                                             b.data()[row * m_a.num_cols + col];
    }
  }

  // multiply the bias
  if(m_a.has_bias) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out.bias()[col] = a.bias()[col] * b.bias()[col];
    }
  }
}

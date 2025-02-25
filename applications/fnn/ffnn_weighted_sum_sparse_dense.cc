#include "ffnn_types.h"
#include "ffnn_weighted_sum_sparse_dense.h"
#include <cassert>

bbts::ffnn_weighted_sum_sparse_dense::ffnn_weighted_sum_sparse_dense() {

  // set the names
  impl_name = "ffnn_weighted_sum_sparse_dense_cpu";
  ud_name = "ffnn_weighted_sum_sparse_dense";

  // set the input and output types
  inputTypes = {"ffnn_sparse", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &ffnn_weighted_sum_sparse_dense::add;
}

size_t bbts::ffnn_weighted_sum_sparse_dense::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_weighted_sum_sparse_dense::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_sparse_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();

  // make sure the 
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the new values
  m_out = { m_a.num_rows, m_a.num_cols, false};
}

void bbts::ffnn_weighted_sum_sparse_dense::add(const bbts::ud_impl_t::tensor_params_t &params,
                                               const bbts::ud_impl_t::tensor_args_t &_in,
                                               bbts::ud_impl_t::tensor_args_t &_out) {

  // get the constants
  auto ca = params.get_float_or_default<0>(1.0f);
  auto cb = params.get_float_or_default<1>(1.0f);

  // get the tensors tensors
  auto &a = _in.get<0>().as<ffnn_sparse_t>();
  auto &b = _in.get<1>().as<ffnn_dense_t>();
  const auto &m_a = a.meta().as<ffnn_sparse_meta_t>().m();
  const auto &m_b = b.meta().as<ffnn_dense_meta_t>().m();

  // make sure the the dimensions match
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);

  // get the output tensor
  auto &out = _out.get<0>().as<ffnn_dense_t>();
  auto &m_out = out.meta().m();

  // add a and b
  for (auto idx = 0; idx < m_a.nnz; ++idx) {
    
    // get the row and column
    auto row = a.data()[idx].row;
    auto col = a.data()[idx].col;

    // sum the value
    out.data()[row * m_a.num_cols + col] = ca * a.data()[idx].val +
                                           cb * b.data()[row * m_b.num_cols + col];
  }

  // set the new meta data
  m_out = {m_a.num_rows, m_a.num_cols, false};
}

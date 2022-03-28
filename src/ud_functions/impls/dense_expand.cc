#include "dense_expand.h"
#include "../../tensor/builtin_formats.h"

#include "../../utils/expand.h"
#include "../../utils/expand_indexer.h"

using namespace bbts;
using namespace utils::expand;

dense_expand_t::dense_expand_t() {
  impl_name = "dense_expand";
  ud_name = "expand";

  inputTypes = {"dense"};
  outputTypes = {"dense"};

  inputInplace = {};

  is_gpu = false;

  fn = &dense_expand_t::f;
}

size_t dense_expand_t::get_complexity_hint(
  const ud_impl_t::tensor_params_t &params,
  const ud_impl_t::meta_args_t &_inn)
{
  // O(n * m)
  const auto &m_a = _inn.get<0>().as<dense_tensor_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

// A helper function to parse the parameters and produce
// an expander that can be used to call expand and compact
row_major_expand_t get_expander(
  uint32_t const& which_input,
  int const& dim_i          ,
  int const& blk_inn_i      ,
  int const& blk_out_i      ,
  int const& which_blk_out_i,
  int const& dim_j          ,
  int const& blk_inn_j      ,
  int const& blk_out_j      ,
  int const& which_blk_out_j)
{
  expand_indexer_t indexer(
    {blk_inn_i, blk_inn_j},
    {blk_out_i, blk_out_j});

  return row_major_expand_t(
    indexer.get_expand_dim(
      {dim_i, dim_j},
      which_input,
      {which_blk_out_i, which_blk_out_j}));
}

row_major_expand_t get_expander(
  const ud_impl_t::tensor_params_t &params)
{
  return get_expander(
    params.get_uint<1>(),
    params.get_uint<2>(),
    params.get_uint<3>(),
    params.get_uint<4>(),
    params.get_uint<5>(),
    params.get_uint<6>(),
    params.get_uint<7>(),
    params.get_uint<8>(),
    params.get_uint<9>());
}

row_major_expand_t get_expander(
  int which_input,
  const ud_impl_t::tensor_params_t &params_without_extra_info)
{
  ud_impl_t::tensor_params_t const& params = params_without_extra_info;
  return get_expander(
    which_input,
    params.get_uint<0>(),
    params.get_uint<1>(),
    params.get_uint<2>(),
    params.get_uint<3>(),
    params.get_uint<4>(),
    params.get_uint<5>(),
    params.get_uint<6>(),
    params.get_uint<7>());
}

void dense_expand_t::get_out_meta(
  ud_impl_t::tensor_params_t const& params,
  ud_impl_t::meta_args_t     const& _inn,
  ud_impl_t::meta_args_t          & _out) const
{
  auto const& m_inn = _inn.get<0>().as<dense_tensor_meta_t>().m();
  auto      & m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // This user defined kernel is suitable for use in a touch op.
  // What this means is that the first two parameters are compact
  // and which_input.
  bool compact         = params.get_bool<0>();
  //uint32_t which_input = params.get_uint<1>(); // this is used in get_expander
  // compact:
  //   If true, then compress this tensor into a smaller version that is more suitable
  //   to sending over the wire. Typically, this is what happens:
  //     given input x -> compact to tmp -> move to another node -> write to output y
  //     x = [1,2,3,4] -> [3,4]          ->                      -> [write 3, write 4]
  //   If false, write directly into the output
  //     x = [1,2,3,4] ->                                           [write 3, write 4]
  // which_input: Running a full touch operation may require multiple calls to the
  //   corresponding touch kernel. This parameter specifies which of the inputs has been given.
  //     x0 = [1,2], x1 = [3,4], y = [*,*,*,*].
  //   Then which_input = 0 means we have x0, so write y = [1,2,*,*]
  //   Then which_input = 1 means we have x1, so write y = [*,*,3,4]
  //
  //
  // There are three types of operations that this kernel is required to do.
  //   1. (        input, compact = false) -> write to corresponding portion of the output
  //   2. (compact input, compact = false) -> write to corresponding portion of the output
  //   3. (        input, compact = true ) -> write to a contiguos tmp
  // If compact is false, it is necessary to parse the parameters to see if the input
  // is a compact input or not.
  //
  // params contain for each dimension, the
  //   relation dimension size, num blocks inn, num blocks out, which output block to deal with

  // get_expander parses the params and produces an expander to make it easy
  row_major_expand_t expander = get_expander(params);

  vector<int> compact_shape = expander.compact_inn_shape();
  vector<int> final_out_shape = expander.expand_out_shape();

  if(compact) {
    m_out = { (uint32_t)compact_shape[0], (uint32_t)compact_shape[1] };
  } else {
    m_out = { (uint32_t)final_out_shape[0], (uint32_t)final_out_shape[1] };
  }
}

//#include <chrono>
//#include <ostream>
//#include <mutex>
//#include <thread>
//using time_measurement_t = decltype(std::chrono::high_resolution_clock::now());
//void cu_debug_write(
//  time_measurement_t start_,
//  time_measurement_t end_,
//  std::string name)
//{
//  static std::mutex m;
//  static std::ofstream file("dense_expand_debug.out");
//
//  // one line is (start, end, thread, name)
//  std::thread::id id = std::this_thread::get_id();
//  auto start =
//    std::chrono::duration_cast<std::chrono::nanoseconds>(
//      start_.time_since_epoch()).count();
//  auto end =
//    std::chrono::duration_cast<std::chrono::nanoseconds>(
//      end_.time_since_epoch()).count();
//
//  std::lock_guard<std::mutex> lock(m);
//  file << start << "," << end << "," << id << "," << name << std::endl;
//  file.flush();
//}
//struct cu_debug_write_t_ {
//  cu_debug_write_t_(std::string name):
//    name(name), start(std::chrono::high_resolution_clock::now())
//  {}
//
//  ~cu_debug_write_t_() {
//    auto end = std::chrono::high_resolution_clock::now();
//    cu_debug_write(start, end, name);
//  }
//
//  std::string name;
//  time_measurement_t start;
//};
//  cu_debug_write_t_ asd(std::to_string((uint64_t)out.data()));

void dense_expand_t::f(
  ud_impl_t::tensor_params_t const& params,
  ud_impl_t::tensor_args_t   const& _inn,
  ud_impl_t::tensor_args_t        & _out)
{
  auto &inn = _inn.get<0>().as<dense_tensor_t>();
  auto &out = _out.get<0>().as<dense_tensor_t>();

  auto &m_inn = inn.meta().m();
  auto &m_out = out.meta().m();


  row_major_expand_t expander = get_expander(params);

  vector<int> compact_shape = expander.compact_inn_shape();
  vector<int> final_out_shape = expander.expand_out_shape();

  bool compact = params.get_bool<0>();
  if(compact) {
    m_out = { (uint32_t)compact_shape[0], (uint32_t)compact_shape[1] };
    expander.compact(inn.data(), out.data());
  } else {
    m_out = { (uint32_t)final_out_shape[0], (uint32_t)final_out_shape[1] };
    if(compact_shape[0] == m_inn.num_rows && compact_shape[1] == m_inn.num_cols) {
      expander.uncompact(inn.data(), out.data());
    } else {
      expander.expand(inn.data(), out.data());
    }
  }
}

// return whether or not this input can be compacted any
bool dense_expand_t::can_compact(
  uint32_t which_input,
  const bbts::ud_impl_t::tensor_params_t &params_without_extra_info,
  const meta_args_t &_inn) const
{
  auto const& m_inn = _inn.get<0>().as<dense_tensor_meta_t>().m();

  row_major_expand_t expander = get_expander(which_input, params_without_extra_info);

  // If the shapes are the same, this is already compact, so return false.
  // Otherwise, the input shape can be made smaller, so compact it.
  vector<int> compact_shape = expander.compact_inn_shape();
  return (compact_shape[0] != m_inn.num_rows) || (compact_shape[1] != m_inn.num_cols);
}


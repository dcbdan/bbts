#include "../cutensor/cu.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace _register_reduction {

using namespace _cutensor_utils;

using op_t = ud_impl_t::ud_impl_callable;

struct params_t {
  castable_op_t bop;
  float alpha;
  modes_t ordering;
};

params_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  params_t ret;
  ret.bop   = castable_op_t(params.get_int<0>());
  ret.alpha = params.get_float<1>();
  for(int i = 2; i != params.num_parameters(); ++i) {
    ret.ordering.push_back(params.get_raw(i).i);
  }
  return ret;
}

template <typename M>
void set_out_meta(
  params_t const& p,
  M const& inn,
  M& out) {
  out.rank = p.ordering.size();
  for(int r = 0; r != out.rank; ++r) {
    out.dims[r] = inn.dims[p.ordering[r]];
  }
}

// Run the same computation
// to verify that the output in ous is correct.
void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  std::cout << "REFERENCE REDUCTION" << std::endl;
  std::string errmsg = "Reduction reference error. ";

  // assuming parse is corract
  params_t p = parse(params);

  cu_shape_t const& meta_inn = ins.get<0>().as<cu_meta_t>().m();
  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();

  float* data_inn       = (float*)(ins.get<0>().as<cu_t>().data());
  float* data_out_check = (float*)(ous.get<0>().as<cu_t>().data());

  dims_t dims_out;
  for(auto m: p.ordering) {
    dims_out.push_back(meta_inn.dims[m]);
  }

  if(meta_out.rank != dims_out.size()) {
    throw std::runtime_error(errmsg + "ranks");
  }
  for(int i = 0; i != dims_out.size(); ++i) {
    if(dims_out[i] == 0 || meta_out.dims[i] != dims_out[i]) {
      throw std::runtime_error(errmsg + "meta_out");
    }
  }

  auto n_out = product(dims_out);
  float* ref  = new float[n_out];
  bool* check = new bool[n_out];
  std::fill(check, check + n_out, false);

  dims_t dims_inn = cu_shape_as_vec(meta_inn);
  auto do_it= [&](dims_t idxs) {
    size_t i = get_offset(dims_inn, idxs);
    size_t o = get_offset_wrt_ordering(dims_inn, idxs, p.ordering);
    float v = data_inn[i];
    if(check[o]) {
      ref[o] = p.bop.scalar_op(ref[o], v);
    } else {
      ref[o] = v;
      check[o] = true;
    }
  };
  for_each_index f(dims_inn);
  f(do_it);

  for(int i = 0; i != n_out; ++i) {
    ref[i] *= p.alpha;
  }

  float err = max_difference(n_out, ref, data_out_check);
  if(err > 0.00001) {
    //std::cout << "bop: " << p.bop.i << std::endl;
    //std::cout << "alpha: " << p.alpha << std::endl;
    //std::cout << "ordering: ";
    //for(auto d: p.ordering){ std::cout << d << " "; } std::cout << std::endl;
    //for(int i = 0; i != n_out; ++i) {
    //  std::cout << ref[i] << ", " << data_out_check[i] << std::endl;
    //}
    throw std::runtime_error(errmsg);
  }

  delete[] ref;
  delete[] check;
}

struct cpu_op {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const
  {
    cu_debug_write_t("reduction");

    params_t p = parse(params);

    cu_shape_t const& meta_inn =  _in.get<0>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    if(maximum(p.ordering) >= meta_inn.rank) {
      throw std::invalid_argument("input rank is not great enough");
    }

    set_out_meta(p, meta_inn, meta_out);

#ifndef CU_REDUCTION_OFF
    float* data_inn_ = (float*)(_in.get<0>().as<cu_t>().data());
    float* data_out = (float*)(_out.get<0>().as<cu_t>().data());

    // ijk->j
    // ordering = [1]
    // ijkl->il                      klij   2301
    // ordering = [0,3]
    // ijkl->li                      jkli   3012
    // ordering = [3,0]

    // permute meta_inn so that it has shape agg modes + output modes
    // so that each output then each agg mode can be traversed in order
    modes_t permutation(meta_inn.rank, -1);
    {
      int rank_agg = meta_inn.rank - meta_out.rank;
      for(int r = 0; r != meta_out.rank; ++r) {
        permutation[p.ordering[r]] = rank_agg + r;
      }
      int j = 0;
      for(int i = 0; i != meta_inn.rank; ++i) {
        if(permutation[i] == -1) {
          permutation[i] = j++;
        }
      }
    }

    permute_t permute(cu_shape_as_vec(meta_inn), permutation, data_inn_);
    float* data_inn = permute.get();

    // now for each output, sum up the items into the result
    int num_out = _out.get<0>().as<cu_meta_t>().num_elem();
    int num_inn = _in.get<0>().as<cu_meta_t>().num_elem();
    int num_agg = num_inn / num_out;
    for(int i = 0; i != num_out; ++i) {
      data_out[i] = data_inn[i*num_agg+0];
      for(int j = 1; j < num_agg; ++j) {
        data_out[i] = p.bop.scalar_op(data_out[i], data_inn[i*num_agg+j]);
      }
    }

    // multiply by alpha
    cblas_sscal(num_out, p.alpha, data_out, 1);

#ifdef CU_BARB_REFERENCE
    reference(params, _in, _out);
#endif
#endif
  }
};

struct f : public ud_impl_t {
  f(std::string name, bool is_gpu_, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = is_gpu_;
    fn     = op;
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    return _in.get<0>().as<cu_meta_t>().num_elem();
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in,
    meta_args_t &_out) const override {

    params_t p = parse(params);

    auto const& meta_inn =  _in.get<0>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    if(maximum(p.ordering) >= meta_inn.rank) {
      throw std::invalid_argument("input rank is not great enough");
    }

    set_out_meta(p, meta_inn, meta_out);
  }
};

}

void register_reduction(
  udf_manager_ptr udf_manager,
  std::string name) {

  // make an f, do the thing.
  udf_manager->register_udf(std::make_unique<ud_func_t>(
    ud_func_t {
        .ud_name = name,
        .is_ass = false,
        .is_com = false,
        .num_in = 1,
        .num_out = 1,
        .impls = {}
    }));

  udf_manager->register_udf_impl(
    std::make_unique<_register_reduction::f>(name, false, _register_reduction::cpu_op()));
}



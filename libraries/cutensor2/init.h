#include "../cutensor/cu.h"
#include "utils.h"

#include <random>
#include <algorithm>

using namespace bbts;

namespace _register_init {

using namespace _cutensor_utils;

struct params_t {
  bool is_random;
  float f1;
  float f2;
  dims_t dims;
};

params_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  params_t ret;
  ret.is_random = params.get_bool<0>();
  int i;
  if(ret.is_random) {
    ret.f1 = params.get_float<1>();
    ret.f2 = params.get_float<2>();
    i = 3;
  } else {
    ret.f1 = params.get_float<1>();
    i = 2;
  }
  for(; i != params.num_parameters(); ++i) {
    ret.dims.push_back(params.get_raw(i).i);
  }
  return ret;
}

template <typename M>
void set_out_meta(
  params_t const& p,
  M& out) {
  out.rank = p.dims.size();
  for(int r = 0; r != out.rank; ++r) {
    out.dims[r] = p.dims[r];
  }
}

struct op {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const {
    params_t p = parse(params);
    set_out_meta(p, _out.get<0>().as<cu_meta_t>().m());
    size_t n = product(p.dims);
    cu_t& out = _out.get<0>().as<cu_t>();
    float* data = (float*)out.data();
    if(p.is_random) {
      // the random device should be somewhere else and thread
      // independent, but this is ok since there is (currently)
      // no need for high quality random numbers
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(p.f1, p.f2);
      std::generate(data, data + n, [&](){ return dis(gen); });
    } else {
      std::fill(data, data + n, p.f1);
    }
  }
};

struct f : public ud_impl_t {
  f(std::string name) {
    impl_name = name;
    ud_name = name;
    inputTypes = {};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = false;
    fn = op();
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    return product(parse(params).dims);
  }

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override {
    set_out_meta(parse(params), _out.get<0>().as<cu_meta_t>().m());
  }
};

}

void register_init(
  udf_manager_ptr udf_manager,
  std::string name) {

  udf_manager->register_udf(std::make_unique<ud_func_t>(
        ud_func_t {
          .ud_name = name,
          .is_ass = false,
          .is_com = false,
          .num_in = 0,
          .num_out = 1,
          .impls = {}
        }));
  udf_manager->register_udf_impl(std::make_unique<_register_init::f>(name));
}


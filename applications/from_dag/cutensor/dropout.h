#include "cu.h"
#include "utils.h"

#include <random>
#include <algorithm>

using namespace bbts;

namespace _register_dropout {

using namespace _cutensor_utils;

template <typename M>
void set_out_meta(
  M const& inn,
  M      & out) {
  out.rank = inn.rank;
  for(int r = 0; r != out.rank; ++r) {
    out.dims[r] = inn.dims[r];
  }
}

struct op {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const
  {
    cu_debug_write_t("dropout");

    auto const& meta_inn =  _in.get<0>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();
    set_out_meta(meta_inn, meta_out);

    float prob_dropped = params.get_float<0>();

    size_t n = _in.get<0>().as<cu_meta_t>().num_elem();
    float* data_inn = (float*)( _in.get<0>().as<cu_t>().data());
    float* data_out = (float*)(_out.get<0>().as<cu_t>().data());

    if(prob_dropped <= 0.0) {
      std::copy(data_out, data_out + n, data_inn);
    } else {
      // the random device should be somewhere else and thread
      // independent, but this is ok since there is (currently)
      // no need for high quality random numbers TODO
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(0.0, 1.0);

      auto unary_op = [&](float const& v) {
        if(dis(gen) <= prob_dropped) {
          return float(0.0);
        } else {
          return v;
        }
      };
      std::transform(data_out, data_out + n, data_inn, unary_op);
    }
  }
};

struct f : public ud_impl_t {
  f(std::string name) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {0};
    is_gpu = false;
    fn = op();
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    return _in.get<0>().as<cu_meta_t>().num_elem();
  }

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override {
    set_out_meta(_in.get<0>().as<cu_meta_t>().m(),
                _out.get<0>().as<cu_meta_t>().m());
  }
};

}

void register_dropout(
  udf_manager_ptr udf_manager,
  std::string name) {

  udf_manager->register_udf(std::make_unique<ud_func_t>(
        ud_func_t {
          .ud_name = name,
          .is_ass = false,
          .is_com = false,
          .num_in = 1,
          .num_out = 1,
          .impls = {}
        }));
  udf_manager->register_udf_impl(std::make_unique<_register_dropout::f>(name));
}

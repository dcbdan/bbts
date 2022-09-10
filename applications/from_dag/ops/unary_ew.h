#include "types.h"

namespace _register_unary_ew {

struct info_t {
  unary_op_t op;
  float alpha;
  int64_t num_elem;
};

info_t parse(bbts::ud_impl_t::tensor_params_t const& params)
{
  info_t ret;
  int i = 0;
  parse_uop(params, ret.op, i); // parse_uop writes to ret.uop and updates i
  ret.alpha = params.get_raw(i++).f;
  ret.num_elem = params.get_raw(i++).i;
  return ret;
}

// Run the same computation
// to verify that the output in ous is correct.
void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  throw std::runtime_error("not implemented");

//  std::cout << "REFERENCE UNARY EW" << std::endl;
//  std::string errmsg = "Unary elementwise reference error. ";
//
//  cu_shape_t const& meta_inn = ins.get<0>().as<cu_meta_t>().m();
//  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();
//
//  // assuming parse is corract
//  info_t p = parse(params, meta_inn);
//
//  float* data_inn = (float*)(ins.get<0>().as<cu_t>().data());
//  float* data_out = (float*)(ous.get<0>().as<cu_t>().data());
//
//  if(meta_inn.rank != meta_out.rank) {
//    throw std::runtime_error(errmsg + "ranks");
//  }
//  for(int i = 0; i != meta_inn.rank; ++i) {
//    if(meta_inn.dims[i] != meta_out.dims[i]) {
//      throw std::runtime_error(errmsg + "meta_out");
//    }
//  }
//
//  if(data_inn == data_out) {
//    // This op is being done in place...
//    // TODO: what checks can be done here?
//    return;
//  }
//
//  if(p.op.i == 7) {
//    // this is dropout, which is random..
//    return;
//  }
//
//  int64_t n = p.num_elem();
//
//  for(int64_t i = 0; i != n; ++i) {
//    float v = data_inn[i];
//    float r = 1.23456;
//    if(p.op.i == 0) {
//      // sigmoid
//      r = std::exp(v);
//      r = r / (1 + r);
//    } else if(p.op.i == 1 ){
//      // exp
//      r = std::exp(v);
//    } else if(p.op.i == 2 ){
//      // Square
//      r = v*v;
//    } else if(p.op.i == 3 ){
//      // Relu
//      r = v > 0.0 ? v : 0.0;
//    } else if(p.op.i == 4 ){
//      // Relu'
//      r = v > 0.0 ? 1.0 : 0.0;
//    } else if(p.op.i == 5 ){
//      // sqrt
//      r = sqrt(v);
//    } else if(p.op.i == 6 ){
//      // add a constant value
//      r = v + p.op.f;
//    } else if(p.op.i == 7) {
//      throw std::runtime_error(errmsg + "can't check dropout");
//    } else {
//      // TODO(not important): implement 8 and 9 ops
//      throw std::runtime_error(errmsg + "no op!");
//    }
//    r *= p.alpha;
//
//    float err = abs(r - data_out[i]);
//    if(err > 0.00001) {
//      throw std::runtime_error(errmsg);
//    };
//  }
}

struct op_t {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous)
  {
    cu_debug_write_t("unary_ew");

    info_t info = parse(params);
    int64_t& size_out = ous.get<0>().as<cu_meta_t>().size();
    size_out = info.num_elem;

    float* data_inn = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_UNARY_EW_OFF
    int64_t const& n = info.num_elem;

    if(info.op.i == 0) {
      // sigmoid
      // (I couldn't find a direct mkl implementation,
      //  and using vsExp is numerically unstable)
      for(int i = 0; i != n; ++i) {
        if(data_inn[i] > 0.0) {
          data_out[i] = 1.0 / (1.0 + std::exp(-1.0*data_inn[i]));
        } else {
          data_out[i] = std::exp(data_inn[i]);
          data_out[i] = data_out[i] / (1.0 + data_out[i]);
        }
      }
      //vsExp(n, data_inn, data_out);
      //for(int i = 0; i != n; ++i) {
      //  data_out[i] = data_out[i] / (1.0 + data_out[i]);
      //}
    } else if(info.op.i == 1 ){
      // exp
      vsExp(n, data_inn, data_out);
    } else if(info.op.i == 2 ){
      // Square
      vsSqr(n, data_inn, data_out);
    } else if(info.op.i == 3 ){
      // Relu
      float zero = 0.0;
      vsFmaxI(n, data_inn, 1, &zero, 0, data_out, 1);
    } else if(info.op.i == 4 ){
      // Relu'
      // (it wasn't immediately obvious how to do this with mkl)
      for(int i = 0; i != n; ++i) {
        data_out[i] = data_inn[i] <= 0.0 ? 0.0 : 1.0;
      }
    } else if(info.op.i == 5 ){
      // sqrt
      vsSqrt(n, data_inn, data_out);
    } else if(info.op.i == 6 ){
      // add a constant value
      vsAddI(n, data_inn, 1, &info.op.f, 0, data_out, 1);
    } else if(info.op.i == 7) {
      float const& prob_dropped = info.op.f;
      if(prob_dropped <= 0.0) {
        std::copy(data_out, data_out + n, data_inn);
      } else {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, time(nullptr));

        // 1. Fill everything with a uniform random number in [0,1.0]
        vsRngUniform(
          VSL_RNG_METHOD_UNIFORM_STD,
          stream,
          n,
          data_out,
          0.0,
          1.0);
        // 2. Traverse, either setting or not
        for(int64_t i = 0; i != n; ++i) {
          data_out[i] = data_out[i] <= prob_dropped ? 0.0 : data_inn[i];
        }
      }
    } else if(info.op.i == 8) {
      // NoOp should explicitly handled
      throw std::runtime_error(
          "NoOp in unary_ew kernel. NoOp explicitly designed to not"
          " have unary ew called--it is only for permutations");
    } else if(info.op.i == 9) {
      // absorb scale by alpha here
      float op_alpha = info.alpha * info.op.f;
      // data_out = op_alpha * data_inn + 0.0 * data_out
      cblas_saxpby(n, op_alpha, data_inn, 1,
                      0.0,      data_out, 1);
    } else {
      assert(false);
    }

    if(info.alpha != 1.0 && info.op.i != 9) {
      cblas_sscal(n, info.alpha, data_out, 1);
    }

#ifdef CU_BARB_REFERENCE
    reference(params, ins, ous);
#endif
#endif
  }
};

struct f: public ud_impl_t {
  f(std::string name, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {0};
    is_gpu = false;
    fn     = op;
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override
  {
    info_t info = parse(params);
    return info.num_elem;
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in,
    meta_args_t &_out) const override
  {
    int64_t& size_out = _out.get<0>().as<cu_meta_t>().size();
    info_t info = parse(params);
    size_out = info.num_elem;
  }
};

}

void register_unary_ew(
  udf_manager_ptr udf_manager,
  std::string name)
{
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
    std::make_unique<_register_unary_ew::f>(name, _register_unary_ew::op_t()));
}


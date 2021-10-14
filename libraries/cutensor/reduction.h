#include "cu.h"

namespace _register_reduction {

using M = std::remove_reference<decltype(cu_meta_t(0).m())>::type;

typedef std::vector<int64_t> dims_t;
typedef std::vector<int>     modes_t;

struct op {
  op(cutensorOperator_t scalar_op, modes_t out) 
    : one(1.0), zero(0.0), scalar_op(scalar_op), out(out) {
    for(int r = 0; r != MAXRANK; ++r) {
      inn[r] = r;
    }
  }

  void set_out_meta(M const& meta_inn, M& meta_out) const {
    meta_out.rank = out.size();
    for(int i = 0; i != out.size(); ++i) {
      meta_out.dims[i] = meta_inn.dims[out[i]];
    }
  }

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins, 
    tensor_args_t &ous) {

    auto const& meta_inn = ins.get<0>().as<cu_meta_t>().m();
    auto      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    set_out_meta(meta_inn, meta_out);

    cutensorTensorDescriptor_t desc_inn;
    handle_error("init inn", cutensorInitTensorDescriptor(
      &params.cutensor_handle,
      &desc_inn,
      meta_inn.rank,
      meta_inn.dims,
      NULL, cutensor_scalar_type, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t desc_out;
    handle_error("init out", cutensorInitTensorDescriptor(
      &params.cutensor_handle,
      &desc_out,
      meta_out.rank,
      meta_out.dims,
      NULL, cutensor_scalar_type, CUTENSOR_OP_IDENTITY));

    void* data_inn = ins.get<0>().as<cu_t>().data();
    void* data_out = ous.get<0>().as<cu_t>().data();

    uint64_t worksize;

    handle_error("get workspace", cutensorReductionGetWorkspace(
      &params.cutensor_handle,
      data_inn, &desc_inn, inn,
      data_out, &desc_out, out.data(),
      data_out, &desc_out, out.data(),
      scalar_op, cutensor_compute_type, 
      &worksize));

    void* work = nullptr;
    if(!misc_cuda_cuda_malloc(&work, worksize)) {
      work = nullptr;
      worksize = 0;
    }

    handle_error(cutensorReduction(
      &params.cutensor_handle,
      (void*)&one,  data_inn, &desc_inn, inn,
      (void*)&zero, data_out, &desc_out, out.data(),
                    data_out, &desc_out, out.data(),
      scalar_op, cutensor_compute_type, 
      work, worksize,
      params.stream));
  }

  float one, zero;
  cutensorOperator_t scalar_op;
  int inn[MAXRANK];
  modes_t out;
};

struct f: public ud_impl_t {
  f(std::string name, cutensorOperator_t scalar_op, modes_t out)
    : op_(scalar_op, out) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = true;
    fn     = op_; // TODO: not really sure what is happening by converting
                  //       this guy to std::function. Maybe this stores
                  //       2 op instances, but that isn't an issue.
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

    op_.set_out_meta(
      _in.get<0>( ).as<cu_meta_t>().m(),
      _out.get<0>().as<cu_meta_t>().m());
  }

  op op_;
};

}

void register_reduction(
  udf_manager_ptr udf_manager,
  std::string name,
  cutensorOperator_t scalar_op, 
  std::vector<int> out) {

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
    std::make_unique<_register_reduction::f>(name, scalar_op, out));
}



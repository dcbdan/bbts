#pragma once

#include "cu.h"
#include "misc_cuda.h"
#include "utils.h"

#include <shared_mutex>

namespace _register_contraction {

using namespace _cutensor_utils;

struct key_t {
  dims_t dims;                         // all of the dimensions
  modes_t m_lhs, m_rhs, m_out;         // mode orderings for inputs and output
  uint32_t ali_lhs, ali_rhs, ali_out;  // the tensor alignment values
  float alpha;
};

template <typename T>
bool compare_vec(std::vector<T> const& lhs, std::vector<T> const& rhs) {
  if(lhs.size() != rhs.size()) {
    return false;
  }
  for(int i = 0; i != lhs.size(); ++i) {
    if(lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

bool operator==(const key_t& x, const key_t& y) {
  return compare_vec(x.dims, y.dims)   &&
         compare_vec(x.m_lhs, y.m_lhs) &&
         compare_vec(x.m_rhs, y.m_rhs) &&
         compare_vec(x.m_out, y.m_out) &&
         x.ali_lhs == y.ali_lhs        &&
         x.ali_rhs == y.ali_rhs        &&
         x.ali_out == y.ali_out;
}

// just makin something up; I don't think typical use cases
// will have much more than 10 items in a map.
struct key_hash {
  size_t operator()(key_t const& k) const {
    size_t out = 1;
    for(auto d: k.dims) {
      out *= d;
    }
    out += 7;
    int i = 1;
    for(auto m: k.m_lhs) {
      out += m*(i++);
    }
    for(auto m: k.m_rhs) {
      out += m*(i++);
    }
    for(auto m: k.m_out) {
      out += m*(i++);
    }
    out += (k.ali_lhs + k.ali_rhs + k.ali_out);
    return out;
  }
};

struct plan_t {
  cutensorContractionPlan_t plan;
  size_t worksize;
};

struct op {
  op(): zero(0.0) {}

  // TODO: should copy over plans too?
  op(const op& other): zero(0.0) {}

  struct plan_t {
    cutensorContractionPlan_t plan;
    size_t worksize;
  };

  key_t build_key_sans_alignment(
    const bbts::ud_impl_t::tensor_params_t &params,
    M const& meta_lhs,
    M const& meta_rhs) const {
    key_t ret;
    auto const& rank_lhs = params.get_int<0>(); ret.m_lhs.reserve(rank_lhs);
    auto const& rank_rhs = params.get_int<1>(); ret.m_rhs.reserve(rank_rhs);
    auto const& rank_out = params.get_int<2>(); ret.m_out.reserve(rank_out);
    int i = 3;
    int m = 0;
    for(; i != 3 + rank_lhs; ++i) {
      ret.m_lhs.push_back(params.get_raw(i).i);
      m = std::max(m, 1 + ret.m_lhs.back());
    }
    for(; i != 3 + rank_lhs + rank_rhs; ++i) {
      ret.m_rhs.push_back(params.get_raw(i).i);
      m = std::max(m, 1 + ret.m_rhs.back());
    }
    for(; i != 3 + rank_lhs + rank_rhs + rank_out; ++i) {
      ret.m_out.push_back(params.get_raw(i).i);
    }
    ret.alpha = params.get_raw(i).f;
    ret.dims = dims_t(m);
    for(int j = 0; j != meta_lhs.rank; ++j) {
      ret.dims[ret.m_lhs[j]] = meta_lhs.dims[j];
    }
    for(int j = 0; j != meta_rhs.rank; ++j) {
      ret.dims[ret.m_rhs[j]] = meta_rhs.dims[j];
    }
    return ret;
  }

  plan_t const& get_plan_set_alpha_and_out(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous,
    float& alpha) {

    // the meta for everything is already set
    auto const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    auto const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    auto&       meta_out = ous.get<0>().as<cu_meta_t>().m();

    key_t key = build_key_sans_alignment(params, meta_lhs, meta_rhs);
    alpha = key.alpha;

    set_out_meta(key, meta_out);

    cutensorTensorDescriptor_t desc_lhs;
    handle_error("init lhs", cutensorInitTensorDescriptor(
      &params.cutensor_handle,
      &desc_lhs,
      meta_lhs.rank,
      meta_lhs.dims,
      NULL, cutensor_scalar_type, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t desc_rhs;
    handle_error("init rhs", cutensorInitTensorDescriptor(
      &params.cutensor_handle,
      &desc_rhs,
      meta_rhs.rank,
      meta_rhs.dims,
      NULL, cutensor_scalar_type, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t desc_out;
    handle_error("init out", cutensorInitTensorDescriptor(
      &params.cutensor_handle,
      &desc_out,
      meta_out.rank,
      meta_out.dims,
      NULL, cutensor_scalar_type, CUTENSOR_OP_IDENTITY));

    void* data_lhs = ins.get<0>().as<cu_t>().data();
    void* data_rhs = ins.get<1>().as<cu_t>().data();
    void* data_out = ous.get<0>().as<cu_t>().data();

    handle_error("ali lhs", cutensorGetAlignmentRequirement(
      &params.cutensor_handle,
      data_lhs,
      &desc_lhs,
      &key.ali_lhs));
    handle_error("ali rhs", cutensorGetAlignmentRequirement(
      &params.cutensor_handle,
      data_rhs,
      &desc_rhs,
      &key.ali_rhs));
    handle_error("ali out", cutensorGetAlignmentRequirement(
      &params.cutensor_handle,
      data_out,
      &desc_out,
      &key.ali_out));

    // get a shared lock and see if you can get a plan
    {
      std::shared_lock lock(mutex_);
      auto iter = plans.find(key);
      if (iter != plans.end()) {
        return iter->second;
      }
    }

    // ok, there wasn't a plan. create a new plan,
    // and insert it once an exclusive lock is had

    cutensorContractionDescriptor_t desc_op;
    handle_error("desc op", cutensorInitContractionDescriptor(
      &params.cutensor_handle,
      &desc_op,
      &desc_lhs, key.m_lhs.data(), key.ali_lhs,
      &desc_rhs, key.m_rhs.data(), key.ali_rhs,
      &desc_out, key.m_out.data(), key.ali_out,
      &desc_out, key.m_out.data(), key.ali_out,
      cutensor_compute_type));

    cutensorContractionFind_t find;
    handle_error("find", cutensorInitContractionFind(
      &params.cutensor_handle,
      &find,
      CUTENSOR_ALGO_DEFAULT));

    plan_t plan;
    handle_error("workspace", cutensorContractionGetWorkspace(
      &params.cutensor_handle,
      &desc_op,
      &find,
      CUTENSOR_WORKSPACE_RECOMMENDED,
      &plan.worksize));

    handle_error("plan", cutensorInitContractionPlan(
      &params.cutensor_handle,
      &plan.plan,
      &desc_op,
      &find,
      plan.worksize));

    std::unique_lock lock(mutex_);
    //if this doesn't suceed, that is because key is already
    //in there
    auto iter = plans.insert({key, plan}).first;
    return iter->second;
  }

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous) {

    float alpha;
    plan_t const& plan = get_plan_set_alpha_and_out(params, ins, ous, alpha);

    void* data_lhs = ins.get<0>().as<cu_t>().data();
    void* data_rhs = ins.get<1>().as<cu_t>().data();
    void* data_out = ous.get<0>().as<cu_t>().data();

    void* work = nullptr;
    size_t worksize = plan.worksize;
    // this is a bit goofy: Instead of making this a .cu file, s
    // misc_cuda.cu creates an api for all the cuda stuff that is
    // needed. And only cudaMalloc is needed.
    if(!misc_cuda_cuda_malloc(&work, worksize)) {
      work = nullptr;
      worksize = 0;
    }

    handle_error("cutensorContraction", cutensorContraction(
      &params.cutensor_handle,
      &plan.plan,
      (void*)&alpha,  data_lhs, data_rhs,
      (void*)&zero, data_out, data_out,
      work, worksize, params.stream));
  }

  void set_out_meta(key_t const& k, M& z) const {
    z.rank = k.m_out.size();
    for(int r = 0; r != z.rank; ++r) {
      z.dims[r] = k.dims[k.m_out[r]];
    }
  }

  size_t get_complexity_hint(key_t const& k) const {
    return product(k.dims);
  }

  float zero;
  std::unordered_map<key_t, plan_t, key_hash> plans;
  // just in case, we want it so that many items can read plans
  // at the same time, but writes can only happen on their own
  mutable std::shared_mutex mutex_;
};

struct f: public ud_impl_t {
  f(std::string name) : op_() {
  impl_name = name;
  ud_name = name;
  inputTypes = {"cutensor", "cutensor"};
  outputTypes = {"cutensor"};
  inputInplace = {};
  is_gpu = true;
  fn     = op_; // TODO: not really sure what is happening by converting
                //       this guy to std::function. Maybe this stores
                //       2 op instances, but that isn't an issue.
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    auto k = op_.build_key_sans_alignment(
               params,
               _in.get<0>().as<cu_meta_t>().m(),
               _in.get<1>().as<cu_meta_t>().m());
    return op_.get_complexity_hint(k);
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in,
    meta_args_t &_out) const override {
    auto k = op_.build_key_sans_alignment(
               params,
               _in.get<0>().as<cu_meta_t>().m(),
               _in.get<1>().as<cu_meta_t>().m());
    op_.set_out_meta(k, _out.get<0>().as<cu_meta_t>().m());
  }

  op op_;
};

}

void register_contraction(
  udf_manager_ptr udf_manager,
  std::string name) {

  // make an f, do the thing.
  udf_manager->register_udf(std::make_unique<ud_func_t>(
    ud_func_t {
        .ud_name = name,
        .is_ass = false,
        .is_com = false,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
    }));
  udf_manager->register_udf_impl(
    std::make_unique<_register_contraction::f>(name));
}


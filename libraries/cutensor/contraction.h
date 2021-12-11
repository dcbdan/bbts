#pragma once

#include "cu.h"
#include "misc_cuda.h"
#include "utils.h"

#include <shared_mutex>

namespace _register_contraction {

using namespace _cutensor_utils;

using op_t = ud_impl_t::ud_impl_callable;

struct info_t {
  dims_t dims;
  modes_t m_lhs, m_rhs, m_out;
  float alpha;
};

struct ali_t {
  uint32_t lhs, rhs, out;
};

info_t parse(
  const bbts::ud_impl_t::tensor_params_t &params,
  cu_shape_t const& meta_lhs,
  cu_shape_t const& meta_rhs)
{
  info_t ret;
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

void set_out_meta(info_t const& k, cu_shape_t& z) {
  z.rank = k.m_out.size();
  for(int r = 0; r != z.rank; ++r) {
    z.dims[r] = k.dims[k.m_out[r]];
  }
}

struct key_t {
  dims_t dims;                         // all of the dimensions
  modes_t m_lhs, m_rhs, m_out;         // mode orderings for inputs and output
  uint32_t ali_lhs, ali_rhs, ali_out;  // the tensor alignment values

  key_t(info_t const& info, ali_t const& ali):
    dims(info.dims),
    m_lhs(info.m_lhs), m_rhs(info.m_rhs), m_out(info.m_out),
    ali_lhs(ali.lhs), ali_rhs(ali.rhs), ali_out(ali.out)
  {}
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

struct cpu_op {
  // The computation is
  //   remove lhs only modes and permute lhs
  //   remove rhs only modes and permute rhs
  //   batched matmul
  //   permute out
  //
  // Batched matmul covers the following multiplications:
  //   ijbl,jkbr->ikb
  //   ij  kj     ik
  //   ji  jk     ik
  //   ji  kj     ik
  //   ij  jk     ki * if t_out, then the left and right
  //   ij  kj     ki * arguments are swapped
  //   ji  jk     ki * and transposed in the batch matmul call
  //   ji  kj     ki *

  //
  struct plan_t {
    modes_t permute_lhs, permute_rhs;
    int nb, ni, nj, nk, nl, nr;
    bool t_lhs, t_rhs, t_out;
    modes_t permute_out;
  };

  struct which_t {
    // you can use bits and stuff, if you want
    bool left;
    bool right;
    bool out;

    which_t(): left(false), right(false), out(false) {}

    void set_left() {
      left = true;
    }
    void set_right() {
      right = true;
    }
    void set_out() {
      out = true;
    }

    bool i() const { return  left && !right &&  out; }
    bool j() const { return  left &&  right && !out; }
    bool k() const { return !left &&  right &&  out; }
    bool l() const { return  left && !right && !out; }
    bool r() const { return !left &&  right && !out; }
    bool b() const { return  left &&  right &&  out; }
  };

  // Ideally..
  //   keep the permutations to be no ops, if possible.
  //   otherwise keep the permutations small.
  //   prefer permuting the output since that can be done in place.
  //     (unless lhs or rhs are small).
  // But in reality:
  //   - convert lhs to
  //       ijbl or jibl
  //   - convert rhs to
  //       jkbr or kjbr
  //   - convert output from ikb to wtvr it was supposed ot be
  //
  static plan_t get_plan(info_t const& info) {
    int rank_inc = info.dims.size();
    int rank_lhs = info.m_lhs.size();
    int rank_rhs = info.m_rhs.size();
    int rank_out = info.m_out.size();

    // for each of the incident modes, add a which object which is used
    // to determine which mode of the mode it is among i,j,k,l,r,b
    std::vector<which_t> which(rank_inc);
    for(auto const& m: info.m_lhs) { which[m].set_left();  }
    for(auto const& m: info.m_rhs) { which[m].set_right(); }
    for(auto const& m: info.m_out) { which[m].set_out();   }

    plan_t plan;

    // initialize the sizes of each mode group in plan
    plan.nb = 1; plan.ni = 1; plan.nj = 1; plan.nk = 1; plan.nl = 1; plan.nr = 1;
    for(int x = 0; x != rank_inc; ++x) {
           if(which[x].i()) { plan.ni *= info.dims[x]; }
      else if(which[x].j()) { plan.nj *= info.dims[x]; }
      else if(which[x].k()) { plan.nk *= info.dims[x]; }
      else if(which[x].l()) { plan.nl *= info.dims[x]; }
      else if(which[x].r()) { plan.nr *= info.dims[x]; }
      else if(which[x].b()) { plan.nb *= info.dims[x]; }
      else { throw std::invalid_argument("should not happen"); }
    }

    // collect the orderings of the groupings that
    // will be imposed...
    modes_t i_grp, j_grp, k_grp, b_grp, l_grp, r_grp;
    for(auto const& m: info.m_out) {
           if(which[m].i()) { i_grp.push_back(m); }
      else if(which[m].k()) { k_grp.push_back(m); }
      else if(which[m].b()) { b_grp.push_back(m); }
    }
    for(auto const& m: info.m_lhs) {
           if(which[m].j()) { j_grp.push_back(m); }
      else if(which[m].l()) { l_grp.push_back(m); }
    }
    for(auto const& m: info.m_rhs) {
      if(which[m].r()) { r_grp.push_back(m); }
    }

    // determine the transpose + permutation of lhs
    {
      modes_t inverse(rank_inc, -1);
      for(int x = 0; x != rank_lhs; ++x) {
        inverse[info.m_lhs[x]] = x;
      }

      plan.t_lhs = false;
      for(auto const& m: info.m_lhs) {
             if(which[m].i()) {                    break; }
        else if(which[m].j()) { plan.t_lhs = true; break; }
      }

      auto add_grp = [&](modes_t const& grp) {
        for(auto const& inc: grp) { plan.permute_lhs.push_back(inverse[inc]); }
      };

      if(plan.t_lhs) { add_grp(j_grp); } else { add_grp(i_grp); }
      if(plan.t_lhs) { add_grp(i_grp); } else { add_grp(j_grp); }
      add_grp(b_grp);
      add_grp(l_grp);
    }

    // determine the transpose + permutation of rhs
    {
      modes_t inverse(rank_inc, -1);
      for(int x = 0; x != rank_rhs; ++x) {
        inverse[info.m_rhs[x]] = x;
      }

      plan.t_rhs = false;
      for(auto const& m: info.m_rhs) {
             if(which[m].j()) {                    break; }
        else if(which[m].k()) { plan.t_rhs = true; break; }
      }

      auto add_grp = [&](modes_t const& grp) {
        for(auto const& inc: grp) {
          plan.permute_rhs.push_back(inverse[inc]);
        }
      };

      if(plan.t_rhs) { add_grp(k_grp); } else { add_grp(j_grp); }
      if(plan.t_rhs) { add_grp(j_grp); } else { add_grp(k_grp); }
      add_grp(b_grp);
      add_grp(r_grp);
    }

    // determine the transpose + permutation for the output
    {
      modes_t inverse(rank_inc, -1);
      for(int x = 0; x != rank_out; ++x) {
        inverse[info.m_out[x]] = x;
      }

      plan.t_out = false;
      for(auto const& m: info.m_out) {
             if(which[m].i()) {                    break; }
        else if(which[m].k()) { plan.t_out = true; break; }
      }

      for(auto const& inc: info.m_out) {
        plan.permute_out.push_back(inverse[inc]);
      }
    }

    return plan;
  }

  struct reduce_t {
    float* inn;
    float* ret;

    ~reduce_t() {
      if(ret != nullptr) {
        delete ret;
      }
    }

    float* get() {
      if(ret == nullptr) {
        return inn;
      } else {
        return ret;
      }
    }

    reduce_t(float* inn_, bool in_place, int nagg, int n): inn(inn_), ret(nullptr) {
      if(nagg > 1) {
        if(in_place) {
          for(int i = 1; i < nagg; ++i) {
            vsAdd(n, inn, inn + n*i, inn);
          }
        } else {
          ret = new float[n];
          std::copy(inn, inn+n, ret);
          for(int i = 1; i < nagg; ++i) {
            vsAdd(n, ret, inn + n*i, ret);
          }
        }
      }
    }
  };

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous)
  {
    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    set_out_meta(info, meta_out);

    float* data_lhs_ = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_rhs_ = (float*)(ins.get<1>().as<cu_t>().data());
    float* data_out  = (float*)(ous.get<0>().as<cu_t>().data());

    plan_t plan = get_plan(info);

    permute_t p_lhs(cu_shape_as_vec(meta_lhs), plan.permute_lhs, data_lhs_);
    permute_t p_rhs(cu_shape_as_vec(meta_rhs), plan.permute_rhs, data_rhs_);

    reduce_t red_lhs(p_lhs.get(), p_lhs.will_delete(), plan.nl, plan.ni*plan.nj*plan.nb);
    reduce_t red_rhs(p_rhs.get(), p_rhs.will_delete(), plan.nr, plan.nj*plan.nk*plan.nb);

    if(!plan.t_out) {
      cblas_sgemm_batch_strided(CblasColMajor,
        plan.t_lhs ? CblasTrans : CblasNoTrans,
        plan.t_rhs ? CblasTrans : CblasNoTrans,
        plan.ni, plan.nk, plan.nj,
        info.alpha,
        red_lhs.get(), plan.t_lhs ? plan.nj : plan.ni, plan.ni*plan.nj,
        red_rhs.get(), plan.t_rhs ? plan.nk : plan.nj, plan.nj*plan.nk,
        0.0,
        data_out, plan.ni, plan.ni*plan.nk,
        plan.nb);
    } else {
      cblas_sgemm_batch_strided(CblasColMajor,
        plan.t_rhs ? CblasNoTrans : CblasTrans,
        plan.t_lhs ? CblasNoTrans : CblasTrans,
        plan.nk, plan.ni, plan.nj,
        info.alpha,
        red_rhs.get(), plan.t_rhs ? plan.nj : plan.nk, plan.nj*plan.nk,
        red_lhs.get(), plan.t_lhs ? plan.ni : plan.nj, plan.ni*plan.nj,
        0.0,
        data_out, plan.ni, plan.ni*plan.nk,
        plan.nb);
    }

    inplace_permute(cu_shape_as_vec(meta_out), plan.permute_out, data_out);
  }
};

struct gpu_op {
  gpu_op(): zero(0.0) {}

  // TODO: should copy over plans too?
  gpu_op(const gpu_op& other): zero(0.0) {}

  struct plan_t {
    cutensorContractionPlan_t plan;
    size_t worksize;
  };

  plan_t const& get_plan_set_alpha_and_out(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous,
    float& alpha) {

    // the meta for everything is already set
    auto const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    auto const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    auto&       meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    ali_t ali;

    alpha = info.alpha;
    set_out_meta(info, meta_out);

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
      &ali.lhs));
    handle_error("ali rhs", cutensorGetAlignmentRequirement(
      &params.cutensor_handle,
      data_rhs,
      &desc_rhs,
      &ali.rhs));
    handle_error("ali out", cutensorGetAlignmentRequirement(
      &params.cutensor_handle,
      data_out,
      &desc_out,
      &ali.out));

    key_t key(info, ali);

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
    tensor_args_t &ous)
  {
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

  float zero;
  std::unordered_map<key_t, plan_t, key_hash> plans;
  // just in case, we want it so that many items can read plans
  // at the same time, but writes can only happen on their own
  mutable std::shared_mutex mutex_;
};

struct f: public ud_impl_t {
  f(std::string name, bool is_gpu_, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor", "cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = is_gpu_;
    fn     = op;
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override
  {
    cu_shape_t const& meta_lhs = _in.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = _in.get<1>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);

    return product(info.dims);
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in,
    meta_args_t &_out) const override
  {
    cu_shape_t const& meta_lhs = _in.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = _in.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    set_out_meta(info, meta_out);
  }
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
    std::make_unique<_register_contraction::f>(
      name, true, _register_contraction::gpu_op()));
  udf_manager->register_udf_impl(
    std::make_unique<_register_contraction::f>(
      name, false, _register_contraction::cpu_op()));
}


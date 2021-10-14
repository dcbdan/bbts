#include "cu.h"
#include "misc_cuda.h"

#include <shared_mutex>

namespace _register_contraction {

using M = std::remove_reference<decltype(cu_meta_t(0).m())>::type;

typedef std::vector<int64_t> dims_t;
typedef std::vector<int>     modes_t;

struct key_t {
  dims_t dims;
  uint32_t lhs, rhs, out;
};

bool operator==(const key_t& x, const key_t& y) {
  // if this is being called, dims is of the same size
  for(int i = 0; i != x.dims.size(); ++i) {
    if(x.dims[i] != y.dims[i]) {
      return false;
    }
  }
  return x.lhs == y.lhs && x.rhs == y.rhs && x.out == y.out;
}

// just makin something up; I don't think typical use cases
// will have much more than 10 items in a map.
struct key_hash {
  size_t operator()(key_t const& k) const {
    size_t out = 0;
    for(int i = 0; i != k.dims.size(); ++i) {
      out += (i+1)*(k.dims[i]+7) % 123456;
    }
    out += k.lhs + k.rhs + k.out;
    return out;
  }
};

struct op {
  op(modes_t lhs, modes_t rhs, modes_t out) 
    : one(1.0), zero(0.0), lhs(lhs), rhs(rhs), out(out) {

    num_dims = 0;
    for(auto const& l: lhs) {
      num_dims = std::max(l + 1, num_dims);
    }
    for(auto const& r: rhs) {
      num_dims = std::max(r + 1, num_dims);
    }
  }

  // whenever this gets copied, the parameter map is reset
  op(const op& other)
    : one(1.0), zero(0.0), lhs(other.lhs), rhs(other.rhs), out(other.out), 
      num_dims(other.num_dims)
  {}

  struct plan_t {
    cutensorContractionPlan_t plan;
    size_t worksize;
  };

  plan_t const& get(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    const tensor_args_t &ous,
    const dims_t& full_dims) {

    // figure out what the key is;
    // the only thing remaning is the tensor alignments
    key_t key;
    key.dims = full_dims;

    // the meta for everything is already set
    auto const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    auto const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    auto const& meta_out = ous.get<0>().as<cu_meta_t>().m();

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
      &key.lhs));
    handle_error("ali rhs", cutensorGetAlignmentRequirement( 
      &params.cutensor_handle,
      data_rhs,
      &desc_rhs,
      &key.rhs));
    handle_error("ali out", cutensorGetAlignmentRequirement( 
      &params.cutensor_handle,
      data_out,
      &desc_out,
      &key.out));

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
      &desc_lhs, lhs.data(), key.lhs,
      &desc_rhs, rhs.data(), key.rhs,
      &desc_out, out.data(), key.out,
      &desc_out, out.data(), key.out,
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
 
  dims_t full_dims(M const& meta_lhs, M const& meta_rhs) const {
    dims_t dims(num_dims);
    for(int i = 0; i != meta_lhs.rank; ++i) {
       dims[lhs[i]] = meta_lhs.dims[i];
     }
     for(int i = 0; i != meta_rhs.rank; ++i) {
       dims[rhs[i]] = meta_rhs.dims[i];
     }
     return dims;
  }
  
  dims_t set_out_meta(M const& x, M const& y, M& z) const {
    dims_t dims = full_dims(x, y);
    z.rank = out.size();
    for(int r = 0; r != z.rank; ++r) {
      z.dims[r] = dims[out[r]];
    }
    return dims;
  }

  size_t get_complexity_hint(M const& x, M const& y) const {
    dims_t dims = full_dims(x, y);
    size_t ret = 1;
    for(int i = 0; i != dims.size(); ++i) {
      ret *= dims[i];
    }
    return ret;
  }

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins, 
    tensor_args_t &ous) {

    // set the out meta
    dims_t full_dims = set_out_meta(
      ins.get<0>().as<cu_meta_t>().m(),
      ins.get<1>().as<cu_meta_t>().m(),
      ous.get<0>().as<cu_meta_t>().m());
  
    plan_t const& plan = get(params, ins, ous, full_dims);
  
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
      (void*)&one,  data_lhs, data_rhs,
      (void*)&zero, data_out, data_out,
      work, worksize, params.stream));
  }

  // these vectors contain integers from 0 to num_dims-1.
  float one, zero;
  modes_t lhs, rhs, out;
  int num_dims;
  // a mapping from num_dims dimension sizes to plans.
  std::unordered_map<key_t, plan_t, key_hash> plans;
  // just in case, we want it so that many items can read plans
  // at the same time, but writes can only happen on their own
  mutable std::shared_mutex mutex_;
};

struct f: public ud_impl_t {
  f(std::string name, modes_t lhs, modes_t rhs, modes_t out)
    : op_(lhs, rhs, out) {
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

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    return op_.get_complexity_hint(
      _in.get<0>( ).as<cu_meta_t>().m(),
      _in.get<1>( ).as<cu_meta_t>().m());
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in, 
    meta_args_t &_out) const override {

    op_.set_out_meta(
      _in.get<0>( ).as<cu_meta_t>().m(),
      _in.get<1>( ).as<cu_meta_t>().m(),
      _out.get<0>().as<cu_meta_t>().m());
  }

  op op_;
};

}

void register_contraction(
  udf_manager_ptr udf_manager,
  std::string name,
  std::vector<int> lhs, 
  std::vector<int> rhs, 
  std::vector<int> out) {

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
    std::make_unique<_register_contraction::f>(name, lhs, rhs, out));
}


#include "cu.h"
#include "utils.h"

#include <mkl.h>
#include <algorithm>
#include <stdexcept>

namespace _register_ewb {

using namespace _cutensor_utils;

using op_t = ud_impl_t::ud_impl_callable;

struct params_t {
  int i;
  float alpha;
  float beta;
  modes_t ordering_lhs, ordering_rhs;
};

params_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  params_t ret;
  ret.i = params.get_int<0>();
  ret.alpha = params.get_float<1>();
  ret.beta  = params.get_float<2>();
  int rank_lhs = params.get_int<3>();
  int i = 4;
  for(; i != 4 + rank_lhs; ++i) {
    ret.ordering_lhs.push_back(params.get_raw(i).i);
  }
  int rank_rhs = params.get_raw(i++).i;
  for(; i != params.num_parameters(); ++i) {
    ret.ordering_rhs.push_back(params.get_raw(i).i);
  }
  return ret;
}

void set_out_meta(
  params_t const& p,
  cu_shape_t const& lhs,
  cu_shape_t const& rhs,
  cu_shape_t& out) {
  out.rank = 1 + std::max(maximum(p.ordering_lhs), maximum(p.ordering_rhs));
  for(int r = 0; r != lhs.rank; ++r) {
    out.dims[p.ordering_lhs[r]] = lhs.dims[r];
  }
  for(int r = 0; r != rhs.rank; ++r) {
    out.dims[p.ordering_rhs[r]] = rhs.dims[r];
  }
}

struct stride_increment_t {
  using stride_op_t = std::function<void(int64_t,int64_t,int64_t)>;

  // COLUMN MAJOR
  stride_increment_t(dims_t const& dims, modes_t const& ordering_lhs, modes_t const& ordering_rhs):
    dims(dims)
  {
    rank = dims.size();

    stride_lhs = dims_t(rank, 0);
    stride_rhs = dims_t(rank, 0);
    stride_out = dims_t(rank, 0);

    auto lhs_iter = ordering_lhs.begin();
    auto rhs_iter = ordering_rhs.begin();

    int64_t l = 1;
    int64_t r = 1;
    int64_t o = 1;
    for(int s = 0; s != rank; ++s) {
      if(lhs_iter != ordering_lhs.end() && *lhs_iter == s) {
        stride_lhs[s] = l;
        l *= dims[s];
        lhs_iter++;
      }
      if(rhs_iter != ordering_rhs.end() && *rhs_iter == s) {
        stride_rhs[s] = r;
        r *= dims[s];
        rhs_iter++;
      }
      stride_out[s] = o;
      o *= dims[s];
    }
  }

  void recurse(stride_op_t op, int64_t s, int64_t lhs, int64_t rhs, int64_t out) {
    if(s == 0) {
      for(int i = 0; i != dims[0]; ++i) {
        op(lhs + i*stride_lhs[0], rhs + i*stride_rhs[0], out + i*stride_out[0]);
      }
    } else {
      for(int i = 0; i != dims[s]; ++i) {
        recurse(op, s-1, lhs + i*stride_lhs[s], rhs + i*stride_rhs[s], out + i*stride_out[s]);
      }
    }
  }

  void operator()(stride_op_t op) {
    if(rank == 0) {
      op(0, 0, 0);
    } else {
      recurse(op, rank-1, 0, 0, 0);
    }
  }

  int rank;
  dims_t dims;
  dims_t stride_lhs;
  dims_t stride_rhs;
  dims_t stride_out;
};

struct cpu_op {
  cpu_op() {}

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const
  {
    params_t p = parse(params);

    cu_shape_t const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs =  _in.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(p, meta_lhs, meta_rhs, meta_out);

    float* _data_lhs = (float*)(_in.get<0>().as<cu_t>().data());
    float* _data_rhs = (float*)(_in.get<1>().as<cu_t>().data());
    float*  data_out = (float*)(_out.get<0>().as<cu_t>().data());

    permute_t v_lhs(cu_shape_as_vec(meta_lhs), p.ordering_lhs, _data_lhs);
    permute_t v_rhs(cu_shape_as_vec(meta_rhs), p.ordering_rhs, _data_rhs);

    // if mode meta_out.rank-1 is in the tensor, increment is 1 else 0
    int inc_lhs = v_lhs.ordering.back() == meta_out.rank-1 ? 1 : 0;
    int inc_rhs = v_rhs.ordering.back() == meta_out.rank-1 ? 1 : 0;

    // get the transposed input data; it'll be deleted on exit if it did an
    // out-of-place transform.
    float* data_lhs = v_lhs.get();
    float* data_rhs = v_rhs.get();

    decltype(vsAddI) *vec_op;
    switch(p.i) {
      case 0: vec_op = vsAddI;  break;
      case 1: vec_op = vsFmaxI; break;
      case 2: vec_op = vsFminI; break;
      case 3: vec_op = vsMulI;  break;
      case 4: vec_op = vsSubI;  break;
      case 5: vec_op = vsDivI;  break;
    }

    if(meta_out.rank == 0) {
      vec_op(
        1,
        data_lhs, 1,
        data_rhs, 1,
        data_out, 1);
    } else if(meta_out.rank == 1) {
      vec_op(
        meta_out.dims[0],
        data_lhs, inc_lhs,
        data_rhs, inc_rhs,
        data_out, 1);
    } else {
      // All dimensions except the last one is handled by the strider.
      // The last dimension is done by mkl.

      dims_t ds(meta_out.rank-1);
      std::copy(meta_out.dims+1, meta_out.dims+meta_out.rank, ds.begin());

      modes_t olhs;
      if(v_lhs.ordering[0] == 0) {
        olhs = v_lhs.ordering;
      } else {
        olhs.resize(v_lhs.ordering.size()-1);
        std::copy(v_lhs.ordering.begin()+1, v_lhs.ordering.end(), olhs.begin());
      }

      modes_t orhs;
      if(v_rhs.ordering[0] == 0) {
        orhs = v_rhs.ordering;
      } else {
        orhs.resize(v_rhs.ordering.size()-1);
        std::copy(v_rhs.ordering.begin()+1, v_rhs.ordering.end(), orhs.begin());
      }

      auto do_it = [&](int64_t lhs, int64_t rhs, int64_t out) {
        vec_op(
          meta_out.dims[0],
          data_lhs + lhs, inc_lhs,
          data_rhs + rhs, inc_rhs,
          data_out + out, 1);
      };
      stride_increment_t strider = stride_increment_t(ds, olhs, orhs);
      strider(do_it);
    }
  }
};

struct gpu_op {
  gpu_op(): zero(0.0) {
    std::iota(ordering_out, ordering_out + MAXRANK, 0);
  }

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const {

    params_t p = parse(params);

    auto const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto const& meta_rhs =  _in.get<1>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(p, meta_lhs, meta_rhs, meta_out);

    void* data_lhs = _in.get<0>().as<cu_t>().data();
    void* data_rhs = _in.get<1>().as<cu_t>().data();
    void* data_out = _out.get<0>().as<cu_t>().data();

    auto create_desc = [&](
        int rank, const int64_t* dims,
        cutensorOperator_t op,
        cutensorTensorDescriptor_t& desc) {
      handle_error("cutensorInitTensorDescriptor create_desc", cutensorInitTensorDescriptor(
          &params.cutensor_handle,
          &desc,
          rank,
          dims,
          NULL,
          cutensor_scalar_type,
          op));
    };

    // Cutensor only supports the binary element operator when the rhs
    // description is the out description... So we have to detect that..
    // And the only way cutensor knows that descriptions are the same is
    // by checking the memory location is the same...
    auto is_outable = [&](modes_t const& ms) {
      if(ms.size() != meta_out.rank) {
        return false;
      }
      for(int r = 0; r != ms.size(); ++r) {
        if(ms[r] != r) {
          return false;
        }
      }
      return true;
    };

    /* The binary operators
        Add 0
        Max 1
        Min 2
        Mul 3
        Sub 4
        Div 5
    */
    cutensorTensorDescriptor_t desc_lhs, desc_rhs, desc_out;

    create_desc(meta_lhs.rank, meta_lhs.dims, CUTENSOR_OP_IDENTITY, desc_lhs);

    if(p.i == 5) {
      create_desc(meta_rhs.rank, meta_rhs.dims, CUTENSOR_OP_RCP, desc_rhs);
    } else {
      create_desc(meta_rhs.rank, meta_rhs.dims, CUTENSOR_OP_IDENTITY, desc_rhs);
    }

    create_desc(meta_out.rank, meta_out.dims, CUTENSOR_OP_IDENTITY, desc_out);

    float alpha = p.alpha;
    float beta  = p.beta;
    if(p.i == 4) {
      beta = -1.0*beta;
    }

    cutensorOperator_t op;
    if(p.i == 0 || p.i == 4) {
      op = CUTENSOR_OP_ADD;
    } else if(p.i == 1) {
      op = CUTENSOR_OP_MAX;
    } else if(p.i == 2) {
      op = CUTENSOR_OP_MIN;
    } else if(p.i == 3 || p.i == 5) {
      op = CUTENSOR_OP_MUL;
    } else  {
      throw std::invalid_argument("invalid binary op!");
    }

    if(is_outable(p.ordering_rhs) && p.i != 5) {
      handle_error("cutensorElementwiseBinary", cutensorElementwiseBinary(
          &params.cutensor_handle,
          (void*)&alpha, data_lhs, &desc_lhs, p.ordering_lhs.data(),
          (void*)&beta,  data_rhs, &desc_out, ordering_out,
                         data_out, &desc_out, ordering_out,
          op, cutensor_scalar_type, params.stream));
    } else if(is_outable(p.ordering_lhs)) {
      handle_error("cutensorElementwiseBinary", cutensorElementwiseBinary(
          &params.cutensor_handle,
          (void*)&beta,  data_rhs, &desc_rhs, p.ordering_rhs.data(),
          (void*)&alpha, data_lhs, &desc_out, ordering_out,
                         data_out, &desc_out, ordering_out,
          op, cutensor_scalar_type, params.stream));
    } else {
      // out = +(op(alpha*lhs, beta*rhs), 0*out)
      // ^ I think cutensor supports this if the first two cases don't work TODO
      handle_error("cutensorElementwieTrinary", cutensorElementwiseTrinary(
          &params.cutensor_handle,
          (void*)&alpha, data_lhs, &desc_lhs, p.ordering_lhs.data(),
          (void*)&beta,  data_rhs, &desc_rhs, p.ordering_rhs.data(),
          (void*)&zero,  data_out, &desc_out, ordering_out,
                         data_out, &desc_out, ordering_out,
          op, CUTENSOR_OP_ADD, cutensor_scalar_type, params.stream));
    }
  }

  float zero;
  int ordering_out[MAXRANK];
};

struct f : public ud_impl_t {
  f(std::string name, bool is_gpu_, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor", "cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = is_gpu_;
    fn     = op;
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    params_t p = parse(params);

    auto const& lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto const& rhs =  _in.get<1>().as<cu_meta_t>().m();

    int rank = 1 + std::max(maximum(p.ordering_lhs), maximum(p.ordering_rhs));
    dims_t dims(rank, 0);
    for(int r = 0; r != lhs.rank; ++r) {
      dims[p.ordering_lhs[r]] = lhs.dims[r];
    }
    for(int r = 0; r != rhs.rank; ++r) {
      dims[p.ordering_rhs[r]] = rhs.dims[r];
    }
    return product(dims);
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &_in,
    meta_args_t &_out) const override {

    params_t p = parse(params);

    auto const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    auto const& meta_rhs =  _in.get<1>().as<cu_meta_t>().m();
    auto      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(p, meta_lhs, meta_rhs, meta_out);
  }

};

}

void register_ewb(
  udf_manager_ptr udf_manager,
  std::string name)
{
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
#ifdef CU_BARB_USE_GPU
  udf_manager->register_udf_impl(
    std::make_unique<_register_ewb::f>(name, true, _register_ewb::gpu_op()));
#endif
#ifdef CU_BARB_USE_CPU
  udf_manager->register_udf_impl(
    std::make_unique<_register_ewb::f>(name, false, _register_ewb::cpu_op()));
#endif
}


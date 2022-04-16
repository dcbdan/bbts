#include "cu.h"
#include "utils.h"

#include <mkl.h>
#include <algorithm>
#include <stdexcept>

#define EWB01(x) // std::cout << x << std::endl

namespace _register_ewb {

using namespace _cutensor_utils;

using op_t = ud_impl_t::ud_impl_callable;

struct params_t {
  int i;
  float alpha;
  float beta;
  modes_t ordering_lhs, ordering_rhs;
};

std::ostream& operator<<(std::ostream& out, params_t const& p) {
  out << "lhs[" << p.ordering_lhs[0];
  for(int i = 1; i < p.ordering_lhs.size(); ++i) {
    out << "," << p.ordering_lhs[i];
  }
  out << "] ";

  out << "rhs[" << p.ordering_rhs[0];
  for(int i = 1; i < p.ordering_rhs.size(); ++i) {
    out << "," << p.ordering_rhs[i];
  }
  out << "] ";

  out << " | i " << p.i << " | alpha " << p.alpha << " | beta " << p.beta;

  return out;
}

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
  out.rank =
      lhs.rank == 0 && rhs.rank== 0
    ? 0
    : 1 + std::max(maximum(p.ordering_lhs), maximum(p.ordering_rhs));
  for(int r = 0; r != lhs.rank; ++r) {
    out.dims[p.ordering_lhs[r]] = lhs.dims[r];
  }
  for(int r = 0; r != rhs.rank; ++r) {
    out.dims[p.ordering_rhs[r]] = rhs.dims[r];
  }
}

struct ewb {
  ewb(vector<int> dims, vector<int> ord_lhs, vector<int> ord_rhs):
    ord_lhs(ord_lhs), ord_rhs(ord_rhs), dims(dims)
  {}

  void operator()(decltype(vsAddI)* vec_op, float* lhs, float* rhs, float* out) {
    if(dims.size() == 0) {
      vec_op(1, lhs, 1, rhs, 1, out, 1);
      return;
    }
    if(dims.size() == 1) {
      vec_op(dims[0],
        lhs, ord_lhs.size() == 1 ? 1 : 0,
        rhs, ord_rhs.size() == 1 ? 1 : 0,
        out, 1);
      return;
    }

    // Consider [0,1,2]
    //          [  1,2]
    // num_shared_dims = 1
    // num_to_copy = dims[0],
    // inc_lhs     = 1,
    // inc_rhs     = 0
    //
    // [0,1,2]
    // [0,  2]
    // num_shared_dims = 1
    // num_to_copy = dims[0],
    // inc_lhs     = 1
    // inc_rhs     = 1
    //
    // [0,1,2,3,4]
    // [    3,4,5]
    // num_shared_dims = 2
    // num_to_copy = dims[0]*dims[1];
    // inc_lhs     = 1
    // inc_rhs     = 0
    //
    // [0,1,2,3,4]
    // [0,1,3,4]
    // num_shared_dims = 2
    // num_to_copy = dims[0]*dims[1]
    // inc_lhs     = 1
    // inc_rhs     = 1

    // Either they share the leading dimension or they do not.

    // If they share leading dimensions, both inc are 1.

    // Otherwise, one inc is 1 and the other is zero

    int num_to_copy = 1;
    int num_shared_dims = 0;
    int inc_lhs = 1;
    int inc_rhs = 1;
    for(int i = 0; i != dims.size(); ++i) {
      if(ord_lhs.size() > i && ord_lhs[i] == i &&
         ord_rhs.size() > i && ord_rhs[i] == i)
      {
        num_to_copy *= dims[i];
        num_shared_dims++;
      } else {
        break;
      }
    }

    if(num_shared_dims == 0) {
      // They did not share the leading dimension, so one of the increments
      // will be zoro

      if(ord_lhs.size() > 0 && ord_lhs[0] == 0) {
        inc_rhs = 0;
        num_shared_dims = ord_rhs[0];
      } else
      if(ord_rhs.size() > 0 && ord_rhs[0] == 0) {
        inc_lhs = 0;
        num_shared_dims = ord_lhs[0];
      } else {
        assert(false);
      }

      num_to_copy = 1;
      for(int i = 0; i != num_shared_dims; ++i) {
        num_to_copy *= dims[i];
      }
    }

    assert(num_shared_dims > 0);

    // At this point num_to_copy, num_shared_dims, inc_lhs and inc_rhs are set.
    // Now cartesian product over each non shared dimension,
    //   compute the offsets,
    //   call the operator
    int offset_out = 0;
    indexer_t indexer(this, num_shared_dims);
    do {
      int offset_lhs = indexer.lhs();
      if(inc_lhs) {
        offset_lhs *= num_to_copy;
      }

      int offset_rhs = indexer.rhs();
      if(inc_rhs) {
        offset_rhs *= num_to_copy;
      }

      EWB01(offset_lhs << ", " << offset_rhs << ", " << offset_out);

      vec_op(num_to_copy,
        lhs + offset_lhs, inc_lhs,
        rhs + offset_rhs, inc_rhs,
        out, 1);

      out += num_to_copy;
      offset_out += num_to_copy;

    } while(indexer.increment());

  };

  // idx stores which index we are currently on.
  // lhs and rhs gets the offset sans leading dimensions
  struct indexer_t {
    indexer_t(ewb* self_, int num_shared_):
      num_shared(num_shared_),
      self(self_),
      idx(self_->dims.size() - num_shared_, 0),
      lhs_beg(self_->ord_lhs.begin()),
      rhs_beg(self_->ord_rhs.begin())
    {
      while(lhs_beg != self->ord_lhs.end() && *lhs_beg < num_shared) {
        lhs_beg++;
      }
      while(rhs_beg != self->ord_rhs.end() && *rhs_beg < num_shared) {
        rhs_beg++;
      }
    }

    int lhs() const {
      return get(lhs_beg, self->ord_lhs.end());
    }

    int rhs() const {
      return get(rhs_beg, self->ord_rhs.end());
    }

    bool increment() {
      for(int i = num_shared; i != self->dims.size(); ++i) {
        if(idx[i - num_shared] + 1 == self->dims[i]) {
          idx[i - num_shared] = 0;
        } else {
          idx[i - num_shared] += 1;
          return true;
        }
      }
      return false;
    }

  private:
    ewb* self;
    int num_shared;
    vector<int> idx;
    vector<int>::iterator lhs_beg;
    vector<int>::iterator rhs_beg;

    int get(vector<int>::iterator iter, vector<int>::iterator end) const {
      auto const& dims = self->dims;

      int p = 1;
      int total = 0;

      for(; iter != end; ++iter) {
        auto const& i = *iter;
        assert(i >= num_shared);

        total += p*idx[i - num_shared];
        p *= dims[i];
      }

      return total;
    }

  };

private:
  // Assumption: these are increasing and
  //             taken together, they include each index 0,...,dims.size()-1.
  vector<int> ord_lhs;
  vector<int> ord_rhs;

  // the dimensions of the output
  vector<int> dims;

};

// Run the same computation
// to verify that the output in ous is correct.
void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  std::cout << "REFERENCE EWB" << std::endl;
  std::string errmsg = "Elementwise binary reference error. ";

  // assuming parse is corract
  params_t p = parse(params);

  cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
  cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();

  float* data_lhs       = (float*)(ins.get<0>().as<cu_t>().data());
  float* data_rhs       = (float*)(ins.get<1>().as<cu_t>().data());
  float* data_out_check = (float*)(ous.get<0>().as<cu_t>().data());

  int rank_out = p.ordering_lhs.size() == 0 && p.ordering_rhs.size() == 0
                   ? 0
                   : 1 + std::max(maximum(p.ordering_lhs), maximum(p.ordering_rhs));
  dims_t dims_out(rank_out, 0);
  if(rank_out != 0) {
    for(int i = 0; i != meta_rhs.rank; ++i) {
      dims_out[p.ordering_rhs[i]] = meta_rhs.dims[i];
    }
    for(int i = 0; i != meta_lhs.rank; ++i) {
      dims_out[p.ordering_lhs[i]] = meta_lhs.dims[i];
    }

    if(meta_out.rank != dims_out.size()) {
      throw std::runtime_error(errmsg + "ranks");
    }
    for(int i = 0; i != dims_out.size(); ++i) {
      if(dims_out[i] == 0 || meta_out.dims[i] != dims_out[i]) {
        throw std::runtime_error(errmsg + "meta_out");
      }
    }
  }

  auto n_out = product(dims_out);
  float* ref = new float[n_out];

  auto do_it= [&](dims_t idxs) {
    size_t l = get_offset_wrt_ordering(dims_out, idxs, p.ordering_lhs);
    size_t r = get_offset_wrt_ordering(dims_out, idxs, p.ordering_rhs);
    size_t o = get_offset(dims_out, idxs);
    float vL = p.alpha*data_lhs[l];
    float vR = p.beta*data_rhs[r];
    if(p.i ==0) {
      ref[o] = vL + vR;
    } else if(p.i == 1) {
      ref[o] = vL > vR ? vL : vR;
    } else if(p.i == 2) {
      ref[o] = vL > vR ? vR : vL;
    } else if(p.i == 3) {
      ref[o] = vL * vR;
    } else if(p.i == 4) {
      ref[o] = vL - vR;
    } else if(p.i == 5) {
      ref[o] = vL / vR;
    } else {
      throw std::runtime_error(errmsg + "no op!");
    }
  };
  for_each_index f(dims_out);
  f(do_it);

  float err = max_difference(n_out, ref, data_out_check);
  if(err > 0.00001) {
    std::cout << "i " << p.i << std::endl;
    std::cout << "alpha " << p.alpha << std::endl;
    std::cout << "beta " << p.beta << std::endl;
    std::cout << "ord_lhs ";
    for(auto d: p.ordering_lhs){ std::cout << d << " "; } std::cout << std::endl;
    std::cout << "ord_rhs ";
    for(auto d: p.ordering_rhs){ std::cout << d << " "; } std::cout << std::endl;
    for(int i = 0; i != n_out; ++i) {
      std::cout << ref[i] << ", " << data_out_check[i] << std::endl;
    }
    throw std::runtime_error(errmsg);
  }

  delete[] ref;
}

struct cpu_op {
  cpu_op() {}

  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const
  {
    cu_debug_write_t("ewb");
    EWB01("ENTERED EWB");

    params_t p = parse(params);

    EWB01(p);

    auto const num_lhs = _in.get<0>().as<cu_meta_t>().num_elem();
    auto const num_rhs = _in.get<1>().as<cu_meta_t>().num_elem();

    cu_shape_t const& meta_lhs =  _in.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs =  _in.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = _out.get<0>().as<cu_meta_t>().m();

    set_out_meta(p, meta_lhs, meta_rhs, meta_out);

#ifndef CU_EWB_OFF
    float* _data_lhs = (float*)(_in.get<0>().as<cu_t>().data());
    float* _data_rhs = (float*)(_in.get<1>().as<cu_t>().data());
    float*  data_out = (float*)(_out.get<0>().as<cu_t>().data());

    permute_t v_lhs(cu_shape_as_vec(meta_lhs), p.ordering_lhs, _data_lhs);
    v_lhs.scale(p.alpha);

    permute_t v_rhs(cu_shape_as_vec(meta_rhs), p.ordering_rhs, _data_rhs);
    v_rhs.scale(p.beta);

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

    vector<int> ds(meta_out.rank);
    std::copy(meta_out.dims, meta_out.dims+meta_out.rank, ds.begin());

    ewb e(ds, v_lhs.ordering, v_rhs.ordering);
    e(vec_op, data_lhs, data_rhs, data_out);

#ifdef CU_BARB_REFERENCE
    reference(params, _in, _out);
#endif
#endif
    EWB01("EXITING EWB");
  }
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

    assert(p.ordering_lhs.size() == meta_lhs.rank);
    assert(p.ordering_rhs.size() == meta_rhs.rank);

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
  udf_manager->register_udf_impl(
    std::make_unique<_register_ewb::f>(name, false, _register_ewb::cpu_op()));
}


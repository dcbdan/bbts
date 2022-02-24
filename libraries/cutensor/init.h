#include "cu.h"
#include "utils.h"

#include <random>
#include <algorithm>

#include <fstream>
#include <stdexcept>

using namespace bbts;

namespace _register_init {

using namespace _cutensor_utils;

struct params_t {
  int which;
  union {
    struct {
      float low;
      float high;
    } random;
    struct {
      float val;
    } constant;
    struct {
      int which;
    } file;
  } t;
  dims_t bid;        // which block this is in the relation
  dims_t dims;       // the dimensions of this block
  dims_t total_dims; // the dimensions of the entire relation
};

struct indexer_t {
  indexer_t(dims_t const& bid, dims_t const& dims, dims_t const& total_dims):
    total_dims(total_dims), idx(total_dims.size(), 0)
  {
    for(int i = 0; i != total_dims.size(); ++i) {
      start_dims.push_back(bid[i]*dims[i]);
      end_dims.push_back((bid[i]+1)*dims[i]);
    }
  }

  bool in_range() const {
    for(int i = 0; i != total_dims.size(); ++i) {
      if(idx[i] >= start_dims[i] && idx[i] < end_dims[i]) {
        // this index is in range
      } else {
        return false;
      }
    }
    return true;
  }

  bool increment() {
    // COLUMN MAJOR
    bool could_increment = false;
    for(int i = 0; i != total_dims.size(); ++i) {
      if(idx[i] + 1 == total_dims[i]) {
        idx[i] = 0;
      } else {
        idx[i] += 1;
        could_increment = true;
        break;
      }
    }
    return could_increment;
  }

  dims_t total_dims, idx, start_dims, end_dims;
};

void load_block(float* data, params_t const& params) {
  std::string const& file = _cutensor_utils::_input_files[params.t.file.which];
  std::ifstream f(file);
  if(!f.is_open()) {
    throw std::runtime_error("couldn't open: " + file);
  }

  // The idea is to walk through the entire file. Whenever the index is in the
  // right range, write it to the output.
  // This could be faster, but correctness is the first goal.
  indexer_t indexer(params.bid, params.dims, params.total_dims);
  float val;
  do {
    f.read((char*)(&val), sizeof(float));
    if(indexer.in_range()) {
      *data++ = val;
    }
  } while(indexer.increment());
}

params_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  params_t ret;
  ret.which = params.get_int<0>();
  int i;
  if(ret.which == 0) {
    ret.t.random.low  = params.get_float<1>();
    ret.t.random.high = params.get_float<2>();
    i = 3;
  } else if(i == 1) {
    ret.t.constant.val = params.get_float<1>();
    i = 2;
  } else {
    ret.t.file.which = params.get_int<1>();
    i = 2;
  }

  while(i < params.num_parameters()) {
    ret.bid.push_back(       params.get_raw(i++).i);
    ret.dims.push_back(      params.get_raw(i++).i);
    ret.total_dims.push_back(params.get_raw(i++).i);
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
    if(p.which == 0) {
      // the random device should be somewhere else and thread
      // independent, but this is ok since there is (currently)
      // no need for high quality random numbers TODO
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(p.t.random.low, p.t.random.high);
      std::generate(data, data + n, [&](){ return dis(gen); });
    } else if(p.which == 1) {
      std::fill(data, data + n, p.t.constant.val);
    } else {
      load_block(data, p);
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


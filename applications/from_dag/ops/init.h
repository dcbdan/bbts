#include "types.h"

#include "../../../src/utils/expand.h"

using namespace bbts;

namespace _register_init {

struct info_t {
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
  vector<int64_t> bid;        // which block this is in the relation
  vector<int64_t> dims;       // the dimensions of this block
  vector<int64_t> total_dims; // the dimensions of the entire relation
};

std::vector<std::string> _input_files {
  "",  // nothing at index 0..
  "/home/ubuntu/data/X.raw",
  "/home/ubuntu/data/Y.raw"
};

void load_block(float* data, info_t const& params) {
  std::string const& file = _input_files[params.t.file.which];
  std::ifstream f(file);
  if(!f.is_open()) {
    throw std::runtime_error("couldn't open: " + file);
  }

  using utils::expand::part_dim_t;

  int rank = params.dims.size();
  std::vector<part_dim_t> part_dims;
  part_dims.reserve(rank);
  for(int r = 0; r != rank; ++r) {
    part_dims.push_back(part_dim_t{
      .start    = static_cast<int>(params.dims[r]*params.bid[r]),
      .interval = static_cast<int>(params.dims[r]),
      .full     = static_cast<int>(params.total_dims[r])
    });
  }

  utils::expand::column_major_expand_t::for_each_offset(
    part_dims,
    [&](int offset, int num_copy) {
      f.seekg(sizeof(float)*offset);
      f.read((char*)data, sizeof(float)*num_copy);
      data += num_copy;
    });
}

info_t parse(const bbts::ud_impl_t::tensor_params_t &params) {
  info_t ret;
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

struct op {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &_in,
    tensor_args_t &_out) const
  {
    cu_debug_write_t("init");

    info_t p = parse(params);
    size_t n = product_dims(p.dims);
    int64_t& size_out = _out.get<0>().as<cu_meta_t>().size();
    size_out = n;

    float* data = (float*)(_out.get<0>().as<cu_t>().data());

#ifndef CU_INIT_OFF
    if(p.which == 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, time(nullptr));

      vsRngUniform(
        VSL_RNG_METHOD_UNIFORM_STD,
        stream,
        (int32_t) (n),
        data,
        p.t.random.low,
        p.t.random.high);

      vslDeleteStream(&stream);
    } else if(p.which == 1) {
      std::fill(data, data + n, p.t.constant.val);
    } else {
      load_block(data, p);
    }
#endif
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
    return product_dims(parse(params).dims);
  }

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override {
    int64_t& size_out = _out.get<0>().as<cu_meta_t>().size();
    size_out = product_dims(parse(params).dims);
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


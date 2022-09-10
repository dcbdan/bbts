#include "types.h"

#include "../../../src/utils/expand.h"
#include "../../../src/utils/expand_indexer.h"

using namespace utils::expand;

namespace _register_expand {

// TODO: will need to use int64_ts.

// return the expander and whether or not this is a compress op
tuple<
  column_major_expand_t,
  bool>
    get_expander(
      bbts::ud_impl_t::tensor_params_t const& params)
{
  bool compact = params.get_bool<0>();
  int which = static_cast<int>(params.get_uint<1>());


  int rank = (params.num_parameters() - 2) / 4;

  vector<int> dims(rank);
  vector<int> blk_inn(rank);
  vector<int> blk_out(rank);
  vector<int> which_blk_out(rank);
  for(int i = 0; i != rank; ++i) {
    dims[i]          = static_cast<int>(params.get_raw(2 + 4*i).u);
    blk_inn[i]       = static_cast<int>(params.get_raw(3 + 4*i).u);
    blk_out[i]       = static_cast<int>(params.get_raw(4 + 4*i).u);
    which_blk_out[i] = static_cast<int>(params.get_raw(5 + 4*i).u);
  }

  expand_indexer_t indexer(blk_inn, blk_out);
  return {
    column_major_expand_t(
      indexer.get_expand_dim(
        dims,
        which,
        which_blk_out)),
    compact
  };
}

struct expand : public ud_impl_t {
  expand(std::string name, std::string my_name) {
    impl_name = name;
    ud_name = my_name;
    inputTypes = {"cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = false;
    fn = &expand::fn_;
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override
  {
    auto [expander, _1] = _register_expand::get_expander(params);
    auto compact_shape = expander.compact_inn_shape();
    return product_ints(compact_shape);
  }

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_inn, meta_args_t &_out) const override
  {
    DCB01("");

    auto [expander, compact] = _register_expand::get_expander(params);

    auto shape =
      compact                      ?
      expander.compact_inn_shape() :
      expander.expand_out_shape()  ;

    int64_t& size_out = _out.get<0>().as<cu_meta_t>().size();
    size_out = 1;
    for(int const& d: shape) {
      size_out *= d;
    }
  }

  // does the work
  static void fn_(const bbts::ud_impl_t::tensor_params_t &params,
                  const tensor_args_t &ins, tensor_args_t &ous)
  {
    cu_debug_write_t("expand");

    auto [expander, compact] = _register_expand::get_expander(params);

    auto compact_shape   = expander.compact_inn_shape();
    auto final_out_shape = expander.expand_out_shape();

    auto const& out_shape = compact ? compact_shape : final_out_shape;

    int64_t const& size_inn = ins.get<0>().as<cu_meta_t>().size();
    int64_t&       size_out = ous.get<0>().as<cu_meta_t>().size();

    size_out = 1;
    for(int const& d: out_shape) {
      size_out *= d;
    }

    int64_t size_compact = 1;
    for(auto const& d: compact_shape) {
      size_compact *= d;
    }

    float* inn_data = (float*)(ins.get<0>().as<cu_t>().data());
    float* out_data = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_EXPAND_OFF
    if(compact) {
      expander.compact(inn_data, out_data);
    } else {
      if(size_inn == size_compact) {
        // If the input is compacted, uncompact it into out
        expander.uncompact(inn_data, out_data);
      } else {
        // Otherwise, just do the whole thing
        expander.expand(inn_data, out_data);
      }
    }
#endif
  }
};

}

void register_expand(
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
  udf_manager->register_udf_impl(
    std::make_unique<_register_expand::expand>(name, name));
}


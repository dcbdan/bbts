#pragma once

#include "../communication/communicator.h"
#include "../tensor/builtin_formats.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include <iostream>
#include <algorithm>

namespace bbts {

class local_reduce_op_t {
public:

  local_reduce_op_t(
    bbts::tensor_factory_t &_factory,
    bbts::storage_t &_storage,
    const std::vector<tid_t> &_inputs,
    const ud_impl_t::tensor_params_t &_params,
    bbts::tid_t _out_tid,
    const bbts::ud_impl_t &_reduce_op);

  // apply this operation
  void apply();

  bbts::tensor_factory_t &_factory;

  bbts::storage_t &_storage;

  const std::vector<tid_t> &_inputs;

  const ud_impl_t::tensor_params_t &_params;

  bbts::tid_t _out_tid;

  const bbts::ud_impl_t &_reduce_op;

  // make empty input arguments
  bbts::tensor_meta_t _out_meta{};
  bbts::ud_impl_t::tensor_args_t _input_tensors;
  bbts::ud_impl_t::tensor_args_t _output_tensor;
  bbts::ud_impl_t::meta_args_t _input_meta;
  bbts::ud_impl_t::meta_args_t _output_meta;

  // the id of the tensor format
  bbts::tfid_t _id;
private:
  void _inplace_apply();
  void _outplace_apply();
};

}

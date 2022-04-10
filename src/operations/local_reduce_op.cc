#include "local_reduce_op.h"

bbts::local_reduce_op_t::local_reduce_op_t(
  bbts::tensor_factory_t &_factory,
  bbts::storage_t &_storage,
  const std::vector<tid_t> &_inputs,
  const ud_impl_t::tensor_params_t &_params,
  bbts::tid_t _out_tid,
  const bbts::ud_impl_t &_reduce_op) :
    _factory(_factory),
    _storage(_storage),
    _inputs(_inputs),
    _params(_params),
    _out_tid(_out_tid),
    _reduce_op(_reduce_op),
    _input_tensors({nullptr, nullptr}),
    _output_tensor({nullptr}),
    _input_meta({nullptr, nullptr}),
    _output_meta({&_out_meta})
{
  assert(_inputs.size() >= 2);

  // get the impl_id of the output
  _id = _factory.get_tensor_ftm(_reduce_op.outputTypes.front());
}

void bbts::local_reduce_op_t::apply()
{
  if(_reduce_op.inputInplace.size() == 0) {
    return _outplace_apply();
  }
  if(_reduce_op.inputInplace.size() == 2) {
    if(_reduce_op.inputInplace[0] == 0 &&
       _reduce_op.inputInplace[1] == 1)
    {
      return _inplace_apply();
    }
    if(_reduce_op.inputInplace[1] == 0 &&
       _reduce_op.inputInplace[0] == 1)
    {
      return _inplace_apply();
    }
  }

  // One could program inplace_left_apply and inplace_right_apply, but I don't
  // anticipate such a use case...
  throw std::runtime_error("If this error is being called, an out of place reduce _could_ be called.\n"
      "Instead, though, see if both input 0 and 1 can be inplace.\n..."
      "Why are the input inplaces not equal to 0 and 1?");
}

void bbts::local_reduce_op_t::_inplace_apply()
{
  size_t output_size;

  {
    tid_t lhs = _inputs[0];
    tid_t rhs = _inputs[1];

    // calculate the size of the output tensor
    _storage.local_transaction({lhs, rhs}, {}, [&](const storage_t::reservation_result_t &res)
    {
      auto l = res.get[0].get();
      auto r = res.get[1].get();

      // how much do we need to allocated
      _input_meta.set<0>(l.tensor->_meta);
      _input_meta.set<1>(r.tensor->_meta);

      // get the meta data
      _reduce_op.get_out_meta(_params, _input_meta, _output_meta);

      // set the output meta
      auto &type = _reduce_op.outputTypes[0];
      _output_meta.get_by_idx(0).fmt_id = _factory.get_tensor_ftm(type);

      // return the size of the tensor
      output_size = _factory.get_tensor_size(_output_meta.get<0>());
    });

    // perform the first kernel
    _storage.local_transaction({lhs, rhs}, {{_out_tid, output_size}}, [&](const storage_t::reservation_result_t &res) {

      // init the output tensor
      auto &out = res.create_or_get[0].get().tensor;
      _factory.init_tensor(out, _out_meta);

      // get the left and right tensor
      auto l = res.get[0].get().tensor;
      auto r = res.get[1].get().tensor;

      // set the input tensors to the function
      _input_tensors.set<0>(*l);
      _input_tensors.set<1>(*r);

      // set the output tensor to the function
      _output_tensor.set<0>(*out);

      // run the function
      _reduce_op.call_ud(_params, _input_tensors, _output_tensor);
    });
  }

  // now perform the rest of the kernels into the output tensor
  for(size_t idx = 2; idx < _inputs.size(); ++idx) {
    tid_t lhs = _inputs[idx];

    _storage.local_transaction({lhs, _out_tid}, {{_out_tid, output_size}},
      [&](const storage_t::reservation_result_t &res)
    {
      // init the output tensor
      auto &out = res.create_or_get[0].get().tensor;

      // get the left and right tensor
      auto l = res.get[0].get().tensor;
      auto r = res.get[1].get().tensor;

      // set the input tensors to the function
      _input_tensors.set<0>(*l);
      _input_tensors.set<1>(*r);

      // set the output tensor to the function
      _output_tensor.set<0>(*out);

      // run the function
      _reduce_op.call_ud(_params, _input_tensors, _output_tensor);
    });
  }
}

////////////////////////////////////////////////////////////////////////////////
void bbts::local_reduce_op_t::_outplace_apply()
{
  // get the first left side
  tid_t lhs = _inputs.front();
  for(size_t idx = 1; idx < _inputs.size(); ++idx) {

    // get the other side
    tid_t rhs = _inputs[idx];

    // calculate the size of the output tensor
    size_t output_size;
    _storage.local_transaction({lhs, rhs}, {}, [&](const storage_t::reservation_result_t &res) {

      auto l = res.get[0].get();
      auto r = res.get[1].get();

      // how much do we need to allocated
      _input_meta.set<0>(l.tensor->_meta);
      _input_meta.set<1>(r.tensor->_meta);

      // get the meta data
      _reduce_op.get_out_meta(_params, _input_meta, _output_meta);

      // set the output meta
      auto &type = _reduce_op.outputTypes[0];
      _output_meta.get_by_idx(0).fmt_id = _factory.get_tensor_ftm(type);

      // return the size of the tensor
      output_size = _factory.get_tensor_size(_output_meta.get<0>());
    });

    // perform the actual kernel
    tid_t out_tid;
    _storage.local_transaction({lhs, rhs}, {{TID_NONE, output_size}}, [&](const storage_t::reservation_result_t &res) {

      // init the output tensor
      auto &out = res.create_or_get[0].get().tensor;
      _factory.init_tensor(out, _out_meta);

      // get the left and right tensor
      auto l = res.get[0].get().tensor;
      auto r = res.get[1].get().tensor;

      // set the input tensors to the function
      _input_tensors.set<0>(*l);
      _input_tensors.set<1>(*r);

      // set the output tensor to the function
      _output_tensor.set<0>(*out);

      // run the function
      _reduce_op.call_ud(_params, _input_tensors, _output_tensor);

      // set the output tid
      out_tid = res.create_or_get[0].get().id;
    });

    // remove additionally every allocated tensor
    if(idx != 1) {
      _storage.remove_by_tid(lhs);
    }

    // set the output as lhs
    lhs = out_tid;
  }

  // assign a tid to the result of the aggregation
  _storage.assign_tid(lhs, _out_tid);
}

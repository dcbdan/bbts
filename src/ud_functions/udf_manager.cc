#include "udf_manager.h"

namespace bbts {

// the matches
udf_matcher::udf_matcher(std::vector<ud_impl_ptr_t> &impls) : _impls(impls) {}

// try to find a match for the via the input strings
// TODO this is most likely going to be slow but for now it is fine
ud_impl_t* udf_matcher::findMatch(const std::vector<std::string> &inputs,
                                  const std::vector<std::string> &outputs,
                                  bool require_gpu) {

  // go through each implementation
  for(const auto &f : _impls) {

    // check the ud is gpu
    if (f->is_gpu != require_gpu) {
      continue;
    }

    // match the input types
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] != f->inputTypes[i]) { continue; }
    }

    // match the output types
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i] != f->outputTypes[i]) { continue; }
    }

    return f.get();
  }

  // return null
  return static_cast<ud_impl_t*>(nullptr);
}

// initializes the UD manager
udf_manager_t::udf_manager_t(tensor_factory_ptr_t _tensor_factory,
                             gpu_scheduler_ptr_t _gpu_scheduler) : _tensor_factory(std::move(_tensor_factory)),
                                                                   _gpu_scheduler(std::move(_gpu_scheduler)) {

  /// 1. matrix summation
  register_udf(get_matrix_add_udf());

  /// 1.1 add the dense implementation
  register_udf_impl(std::make_unique<dense_matrix_add_t>());

  // check if the gpu is enabled
  if constexpr(static_config::enable_gpu) {

    /// 1.2 add the gpu dense implementation
    register_udf_impl(std::make_unique<dense_matrix_gpu_add_t>());
  }

  /// 2. matrix multiplication
  register_udf(get_matrix_mult_udf());

  /// 2.1 add the dense implementation
  register_udf_impl(std::make_unique<dense_matrix_mult_t>());

  // check if the gpu is enabled
  if constexpr(static_config::enable_gpu) {
    /// 2.2 register the gpu dense implementation
    register_udf_impl(std::make_unique<dense_matrix_gpu_mult_t>());
  }

  /// 3. matrix multiplication
  register_udf(get_matrix_uniform_udf());

  /// 3.1 add the dense implementation
  register_udf_impl(std::make_unique<dense_uniform_t>());
}

// registers a udf with the system
ud_id_t udf_manager_t::register_udf(ud_func_ptr_t _fun) {
  // check if the udf already exists?
  ud_id_t id = _udfs.size();
  auto it = _udfs_name_to_id.find(_fun->ud_name);
  if(it != _udfs_name_to_id.end()) {
    return -1;
  }

  // set the impl_id to the udf
  _fun->id = id;

  // store the function so that it is registered
  _udfs_name_to_id[_fun->ud_name] = id;
  _udfs.emplace_back(std::move(_fun));

  // return the impl_id
  return id;
}

// registers the implementation for a udf with the system
ud_impl_id_t udf_manager_t::register_udf_impl(ud_impl_ptr_t _impl) {
  // if this is a gpu function we need to wrap it so that it runs it on the scheduler
  if(_impl->is_gpu) {

    _impl->gpu_fn = _impl->fn;
    _impl->fn = [&, me = _impl.get()](const bbts::ud_impl_t::tensor_params_t &params,
                                      const bbts::ud_impl_t::tensor_args_t &_in,
                                      bbts::ud_impl_t::tensor_args_t &_out) {

      auto ret = _gpu_scheduler->execute_kernel(me, &params, &_in, &_out);
      ret.get();
    };
  }

  // check if udf is registered.
  auto it = _udfs_name_to_id.find(_impl->ud_name);
  if(it == _udfs_name_to_id.end()) {
    throw std::runtime_error("did you register the udf before registering this impl?");
    return {-1, -1};
  }

  // check if the number of arguments matches
  if(_impl->inputTypes.size() != _udfs[it->second]->num_in &&
      _impl->outputTypes.size() != _udfs[it->second]->num_out)
  {
    throw std::runtime_error("this impl does not have the correct type");
    return {-1, -1};
  }

  // return the impl_id
  return _udfs[it->second]->add_impl(std::move(_impl));
}

// returns a matcher for the given ud function that will be used to figure out
udf_matcher_ptr udf_manager_t::get_matcher_for(const std::string &ud_name) {

  // check if udf is registered.
  auto it = _udfs_name_to_id.find(ud_name);
  if(it == _udfs_name_to_id.end()) {
    return nullptr;
  }

  // make a matcher with the given implementations
  return std::make_unique<udf_matcher>(_udfs[it->second]->impls);
}

udf_matcher_ptr udf_manager_t::get_matcher_for(const std::string &ud_name, bool is_commutative, bool is_associative) {

  // check if udf is registered.
  auto it = _udfs_name_to_id.find(ud_name);
  if(it == _udfs_name_to_id.end()) {
    return nullptr;
  }

  // check for the parameters
  auto &fn = _udfs[it->second];
  if((!fn->is_ass && is_associative) || (!fn->is_com && is_commutative)) {
    return nullptr;
  }

  // make a matcher with the given implementations
  return std::make_unique<udf_matcher>(fn->impls);
}

// return the implementation
ud_impl_t* udf_manager_t::get_fn_impl(ud_impl_id_t _id) {

    // check if we have the ud function
  if(_id.ud_id >= _udfs.size() ||
      _id.impl_id >= _udfs[_id.ud_id]->impls.size()) {
    return nullptr;
  }

  // return the function
  return _udfs[_id.ud_id]->impls[_id.impl_id].get();
}

}

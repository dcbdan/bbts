#include "cu.h"

namespace _register_expand {
  template <typename M>
  void set_out_meta_expand(
    const bbts::ud_impl_t::tensor_params_t &params,
    M& ou) {

    ou.rank = params.num_parameters() / 5;
    for(int idx = 0; idx != ou.rank; ++idx) {
      ou.dims[idx] = params._params[5*idx+1].i;
    }
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
                               const meta_args_t &_in) override {
      int rank = params.num_parameters() / 5;
      int total = 1;
      for(int idx = 0; idx != rank; ++idx) {
        total *= params._params[5*idx+1].i;
      }
      return total;
    }
  
    // return the meta of the output
    void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                      const meta_args_t &_in, meta_args_t &_out) const override {
      set_out_meta_expand(
        params,
        _out.get<0>().as<cu_meta_t>().m());
    }
  
    // does the work
    static void fn_(const bbts::ud_impl_t::tensor_params_t &params,
                    const tensor_args_t &ins, tensor_args_t &ous) {

      auto& meta_out = ous.get<0>().as<cu_meta_t>();
      set_out_meta_expand(params, meta_out.m());
      
      float* data_in = (float*)(ins.get<0>().as<cu_t>().data());
      float* data_ou = (float*)(ous.get<0>().as<cu_t>().data());

      size_t n = meta_out.num_elem();

      // set everything to zero
      memset((void*)data_ou, 0, meta_out.get_data_size() / sizeof(char));
    
      indexer_t idx(params); 
      do {
          (data_ou)[idx.index_out()] = (data_in)[idx.index_in()];
      } while(idx.increment());
    }
  
    struct indexer_t { 
      indexer_t(const bbts::ud_impl_t::tensor_params_t &params) {
        // the parameters store a list of 
        //   [modeSizeIn, modeSizeOut, num, startIn, startOut]
        int num_modes = params.num_parameters() / 5;
        for(int i = 0; i != num_modes; ++i) {
          idx.push_back(0);
          sz_in.push_back    (params._params[5*i + 0].i);
          sz_out.push_back   (params._params[5*i + 1].i);
          num.push_back      (params._params[5*i + 2].i);
          start_in.push_back (params._params[5*i + 3].i);
          start_out.push_back(params._params[5*i + 4].i);
        }
      }
      int index_(std::vector<int> const& start, std::vector<int> const& sz) const {
        // COLUMN MAJOR.
        // The first indices "move the fastest"
        // i0 + s0*i1 + s0*s1*i2 + ... +  s0*...*s(n-1)*in
        // p0*i0 + p1*i1 + p2*i2 + ... + pn*in
        //
        // Here, s[idx] is given by sz[idx],
        // and   i[idx] is given by (start[i] + idx[i])
        int p = 1;
        int total = 0;
        for(int i = 0; i != idx.size(); ++i) {
          total += p*(start[i] + idx[i]);
          p *= sz[i];    
        }
        return total;
      }
      int index_in() const { 
        return index_(start_in, sz_in);
      }
      int index_out() const { 
        return index_(start_out, sz_out);
      }
      // the idea is to iterate through
      // num[0] x num[1] x ... x num[m]
      // The algorithm:
      // - increment from in to back
      // - when you can't increment, set to zero go back 1
      // 0 0
      // 1 0
      // 0 1
      // 1 1
      // 0 2
      // 1 2
      // -----
      // 0 0 0
      // 1 0 0
      // 0 1 0
      // 1 1 0
      // 0 0 1
      // .....
      // 1 1 1
      bool increment() {
        bool could_increment = false;
        for(int i = 0; i != num.size(); ++i) {
          if(idx[i] + 1 == num[i]) {
            idx[i] = 0;
          } else {
            idx[i] += 1;
            could_increment = true;
            break;
          }
        }
        return could_increment;
      }
    
      std::vector<int> idx;
      std::vector<int> sz_in;
      std::vector<int> sz_out;
      std::vector<int> num;
      std::vector<int> start_in;
      std::vector<int> start_out;
    };
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


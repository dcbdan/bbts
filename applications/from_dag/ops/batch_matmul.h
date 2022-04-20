#include "types.h"

namespace _register_batch_matmul {

// This computation covers
//   ijb,jkb->ikb
//   jib,jkb->ikb
//   ijb,kjb->ikb
//   jib,kjb->ikb

struct info_t {
  bool t_lhs;
  bool t_rhs;
  float alpha;

  vector<int64_t> i;
  vector<int64_t> j;
  vector<int64_t> k;
  vector<int64_t> b;

  vector<int64_t> dims() const {
    vector<int64_t> ret;
    ret.reserve(b.size() + i.size() + k.size());

    for(auto& d: i) { ret.push_back(d); }
    for(auto& d: k) { ret.push_back(d); }
    for(auto& d: b) { ret.push_back(d); }

    return ret;
  }

  int64_t ni() const { return product_dims(i); }
  int64_t nj() const { return product_dims(j); }
  int64_t nk() const { return product_dims(k); }
  int64_t nb() const { return product_dims(b); }
};

info_t parse(
  bbts::ud_impl_t::tensor_params_t const& params,
  cu_shape_t const& meta_lhs,
  cu_shape_t const& meta_rhs)
{
  // The params contains
  //   t_lhs, t_rhs, alpha, ni, nj, nk, nb

  info_t ret;

  assert(params.num_parameters() == 7);

  ret.t_lhs = params.get_int<0>();
  ret.t_rhs = params.get_int<1>();
  ret.alpha = params.get_int<2>();

  int ni = params.get_int<3>(); ret.i.reserve(ni);
  int nj = params.get_int<4>(); ret.j.reserve(nj);
  int nk = params.get_int<5>(); ret.k.reserve(nk);
  int nb = params.get_int<6>(); ret.b.reserve(nb);

  assert(ni + nj + nb == meta_lhs.rank);
  assert(nj + nk + nb == meta_rhs.rank);

  { // B LHS AND RHS
    int sb_lhs = ni + nj;
    int sb_rhs = nj + nk;
    for(int x = 0; x != nb; ++x) {
      assert(meta_lhs.dims[sb_lhs + x] == meta_rhs.dims[sb_rhs + x]);
      ret.b.push_back(meta_lhs.dims[sb_lhs+x]);
    }
  }

  { // I LHS
    int si = ret.t_lhs ? nj : 0;
    for(int x = 0; x != ni; ++x) {
      ret.i.push_back(meta_lhs.dims[si + x]);
    }
  }

  { // J LHS AND RHS
    int sj_lhs = ret.t_lhs ? 0 : ni;
    int sj_rhs = ret.t_rhs ? nk : 0;
    for(int x = 0; x != nj; ++x) {
      assert(meta_lhs.dims[sj_lhs + x] == meta_rhs.dims[sj_rhs + x]);
      ret.j.push_back(meta_lhs.dims[sj_lhs+x]);
    }
  }

  { // K RHS
    int sk = ret.t_rhs ? 0 : nj;
    for(int x = 0; x != nk; ++x) {
      ret.k.push_back(meta_rhs.dims[sk + x]);
    }
  }

  return ret;
}

void set_out_meta(info_t info, cu_shape_t& meta_out)
{
  auto dims = info.dims();
  meta_out.rank = dims.size();
  for(int i = 0; i != dims.size(); ++i) {
    meta_out.dims[i] = dims[i];
  }
}

void reference(
  const bbts::ud_impl_t::tensor_params_t &params,
  const tensor_args_t &ins,
  const tensor_args_t &ous)
{
  std::cout << "REFERENCE BATCH MATMUL EW" << std::endl;
  std::string errmsg = "batchmatmul reference error. ";

  cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
  cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
  cu_shape_t const& meta_out = ous.get<0>().as<cu_meta_t>().m();

  float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
  float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
  float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

  info_t info = parse(params, meta_lhs, meta_rhs);

  int di = info.i.size();
  int dj = info.j.size();
  int dk = info.k.size();
  int db = info.b.size();

  for(int x = 0; x != db; ++x) {
    assert(meta_lhs.dims[x + di + dj] == meta_rhs.dims[x + dj + dk]);
    assert(meta_lhs.dims[x + di + dj] == meta_out.dims[x + di + dk]);
  }
  for(int x = 0; x != dj; ++x) {
    assert(meta_lhs.dims[x + (info.t_lhs ? 0 : di)] ==
           meta_rhs.dims[x + (info.t_rhs ? dk : 0)]);
  }

  int64_t nb = info.nb();
  int64_t ni = info.ni();
  int64_t nj = info.nj();
  int64_t nk = info.nk();

  int64_t si_lhs = info.t_lhs ? nj : 1;
  int64_t sj_lhs = info.t_lhs ? 1 : ni;
  int64_t sb_lhs = ni * nj;

  int64_t sj_rhs = info.t_rhs ? nk : 1;
  int64_t sk_rhs = info.t_rhs ? 1 : nj;
  int64_t sb_rhs = nk * nj;

  int64_t si_out = 1;
  int64_t sk_out = ni;
  int64_t sb_out = ni*nk;

  for(int64_t b = 0; b != nb; ++b) {
  for(int64_t i = 0; i != ni; ++i) {
  for(int64_t k = 0; k != nk; ++k) {
    float v = 0.0;
    for(int64_t j = 0; j != nj; ++j) {
      v += data_lhs[si_lhs*i + sj_lhs*j + sb_lhs*b] *
           data_rhs[sj_rhs*j + sk_rhs*k + sb_rhs*b] ;
    }
    v *= info.alpha;

    float err = std::abs(data_out[si_out*i + sk_out*k + sb_out*b] - v);
    if(err > 0.00001) {
      std::cout << v << ", " << data_out[si_out*i + sk_out*k + sb_out*b] << std::endl;
      throw std::runtime_error(errmsg);
    };
  }}}
}

struct op_t {
  void operator()(
    const bbts::ud_impl_t::tensor_params_t &params,
    const tensor_args_t &ins,
    tensor_args_t &ous)
  {
    cu_debug_write_t("batch_matmul");

    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    set_out_meta(info, meta_out);

    float* data_lhs = (float*)(ins.get<0>().as<cu_t>().data());
    float* data_rhs = (float*)(ins.get<1>().as<cu_t>().data());
    float* data_out = (float*)(ous.get<0>().as<cu_t>().data());

#ifndef CU_BATCH_MATMUL_OFF
    bool const& t_lhs = info.t_lhs;
    bool const& t_rhs = info.t_rhs;

    int64_t nb = info.nb();
    int64_t ni = info.ni();
    int64_t nj = info.nj();
    int64_t nk = info.nk();

    cblas_sgemm_batch_strided(CblasColMajor,
      t_lhs ? CblasTrans : CblasNoTrans,
      t_rhs ? CblasTrans : CblasNoTrans,
      ni, nk, nj,
      info.alpha,
      data_lhs,     t_lhs ? nj : ni,    ni*nj,
      data_rhs,     t_rhs ? nk : nj,    nj*nk,
      0.0,
      data_out,     ni,                 ni*nk,
      nb);

#ifdef CU_BARB_REFERENCE
//    reference(params, ins, ous);
#endif

#endif
  }
};

struct f: public ud_impl_t {
  f(std::string name, op_t op) {
    impl_name = name;
    ud_name = name;
    inputTypes = {"cutensor", "cutensor"};
    outputTypes = {"cutensor"};
    inputInplace = {};
    is_gpu = false;
    fn     = op;
  }

  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &ins) override
  {
    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);

    int64_t nb = info.nb();
    int64_t ni = info.ni();
    int64_t nj = info.nj();
    int64_t nk = info.nk();

    return nb*ni*nj*nk;
  }

  void get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const meta_args_t &ins,
    meta_args_t &ous) const override
  {
    cu_shape_t const& meta_lhs = ins.get<0>().as<cu_meta_t>().m();
    cu_shape_t const& meta_rhs = ins.get<1>().as<cu_meta_t>().m();
    cu_shape_t      & meta_out = ous.get<0>().as<cu_meta_t>().m();

    info_t info = parse(params, meta_lhs, meta_rhs);
    set_out_meta(info, meta_out);

    DCB01("lhs,rhs,out: " << meta_lhs << "," << meta_rhs << ", " << meta_out);
  }
};

}

void register_batch_matmul(
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
    std::make_unique<_register_batch_matmul::f>(name, _register_batch_matmul::op_t()));
}

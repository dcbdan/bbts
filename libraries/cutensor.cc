#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/ud_functions/ud_function.h"
#include "../src/ud_functions/udf_manager.h"

#include "cutensor/cu.h"
#include "cutensor/init.h"
#include "cutensor/elementwise.h"
#include "cutensor/misc.h"
#include "cutensor/contraction.h"
#include "cutensor/reduction.h"

// TODO:
//   expand cpu
//   expand gpu
//   cpu add

extern "C" {

  void register_tensors(tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("cutensor", cu_t::get_creation_fs());
  }
  
  void register_udfs(udf_manager_ptr udf_manager) {
    register_init(udf_manager, "init_zero",  0.0);
    register_init(udf_manager, "init_one",   1.0);
    register_init(udf_manager, "init_two",   2.0);
    register_init(udf_manager, "init_three", 3.0);
    register_init_random(udf_manager, "init_random", 0.0, 1.0);

    register_ewb_same_shape(udf_manager, "add", CUTENSOR_OP_ADD, 1.0,  1.0, true, true);
    register_ewb_same_shape(udf_manager, "sub", CUTENSOR_OP_ADD, 1.0, -1.0, true, false);
    register_ewb_same_shape(udf_manager, "max", CUTENSOR_OP_MAX, 1.0,  1.0, true, true);
    register_ewb_same_shape(udf_manager, "min", CUTENSOR_OP_MIN, 1.0,  1.0, true, true);
    register_ewb_same_shape(udf_manager, "mul", CUTENSOR_OP_MUL, 1.0,  1.0, true, true);

    // Note: register_ewb_ij_i is only valid for commutative scalar ops.
    register_ewb_ij_i(udf_manager, "sub_ij_i", CUTENSOR_OP_ADD, 1.0, -1.0);
    register_ewb_ij_i(udf_manager, "max_ij_i", CUTENSOR_OP_MAX, 1.0, 1.0);

    register_ew_same_shape(udf_manager, "sigmoid",  CUTENSOR_OP_SIGMOID, 1.0);
    register_ew_same_shape(udf_manager, "exp",      CUTENSOR_OP_EXP,     1.0);
    register_ew_same_shape(udf_manager, "relu",     CUTENSOR_OP_RELU,    1.0);

    // Misc ones since cutensor has a woefully incomplete selection of scalar ops
    register_reluderiv(udf_manager, "reluderiv");
    register_square(   udf_manager, "square");
    register_div_ij_i( udf_manager, "div_ij_i");

    // contractions
    register_contraction(udf_manager, "matmul",   {0,1}, {1,2}, {0,2});
    register_contraction(udf_manager, "matmulT_", {1,0}, {1,2}, {0,2});
    register_contraction(udf_manager, "matmul_T", {0,1}, {2,1}, {0,2});
    register_contraction(udf_manager, "matmulTT", {1,0}, {2,1}, {0,2});

    // reductions
    register_reduction(udf_manager, "sum_ij_to_i", CUTENSOR_OP_ADD, {0,1}, {0});
    register_reduction(udf_manager, "max_ij_to_i", CUTENSOR_OP_MAX, {0,1}, {0});
  }
}

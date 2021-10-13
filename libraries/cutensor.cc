#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/ud_functions/ud_function.h"
#include "../src/ud_functions/udf_manager.h"

#include "cutensor/cu.h"
#include "cutensor/init.h"
#include "cutensor/elementwise.h"

// TODO:
//   expand
//   register_contraction               # size dependent    (matmuls)
//   register_reduction                 # size dependent    (max_ij_i)
//   register_elementwise               # Just op dependent (relu, relu')
//   register_elementwise_binary        # Just op dependent (add, max)
//
// Need add and expand!
// Need a factory that will create wtvr is necessary!

extern "C" {

  void register_tensors(tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("cutensor", cu_t::get_creation_fs());
  }
  
  void register_udfs(udf_manager_ptr udf_manager) {
    register_init(udf_manager, "init_zero",  0.0);
    register_init(udf_manager, "init_one",   1.0);
    register_init(udf_manager, "init_two",   2.0);
    register_init(udf_manager, "init_three", 3.0);

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

  //
  //  // TODO
  //  // register_ewb_same_shape
  //  // register_ew_same_shape
  //  // register_ewb
  //  //
  //  // register_matmul(bool, bool, size0, size1)
  //  // register_contraction(Op, shape_in, shape_out, size_in)

  //  // register_reluderiv (sigmoid and then ceiling)
  //  // register_square    (recipricoal, abs, sqrt, recipricol)
  //  // register_div_ij_i  (recipricol second...)
  }
}

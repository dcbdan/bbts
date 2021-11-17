#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/ud_functions/ud_function.h"
#include "../src/ud_functions/udf_manager.h"

#include "cutensor/cu.h"
#include "cutensor2/contraction.h"
#include "cutensor2/init.h"
#include "cutensor/expand.h"
#include "cutensor2/reduction.h"

extern "C" {

  void register_tensors(tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("cutensor", cu_t::get_creation_fs());
  }

  void register_udfs(udf_manager_ptr udf_manager) {
    register_init(udf_manager, "init");
    register_expand(udf_manager, "expand");
    register_contraction(udf_manager, "contraction");
    register_reduction(udf_manager, "reduction");
    //register_ew(udf_manager, "ew");
    //register_ewb(udf_manager, "ewb");
    //register_ewb_castable(udf_manager, "ewb");
    //register_dropout(udf_manager, "dropout");
  }

}

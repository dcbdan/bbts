#include "types.h"

#include "permute.h"
#include "unary_ew.h"
#include "binary_ew.h"
#include "castable_ew.h"
#include "batch_matmul.h"
#include "reduction.h"
#include "init.h"
#include "expand.h"

extern "C" {

  void register_tensors(tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("cutensor", cu_t::get_creation_fs());
  }

  void register_udfs(udf_manager_ptr udf_manager)
  {
    register_permute(         udf_manager, "permute"             );
    register_unary_ew(        udf_manager, "unary_ew"            );
    register_binary_ew(       udf_manager, "binary_ew"           );
    register_castable_ew(     udf_manager, "castable_ew"         );
    register_batch_matmul(    udf_manager, "batch_matmul"        );
    register_reduction(       udf_manager, "reduction"           );
    register_init(            udf_manager, "init"                );
    register_expand(          udf_manager, "expand"              );
  }
}

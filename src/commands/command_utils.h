#pragma once

#include "../utils/raw_vector.h"

namespace bbts {

  // a command parameter used to parametrize the kernel calls
  union command_param_t {
    float f;
    int32_t i;
    uint32_t u;
    bool b;
  };

  // the list of parameters
  using command_param_list_t = raw_vector_t<command_param_t>;
}

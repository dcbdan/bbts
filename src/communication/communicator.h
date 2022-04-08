#pragma once

#include "../server/static_config.h"

#ifdef ENABLE_IB
#include "ib_communicator.h"
#else
#include "mpi_communicator.h"
#endif

#include <cstddef>
#include <memory>

namespace bbts {

// TODO: make this file use std::conditional
// and static_config::enable_ib like storage does..
// good luck

#ifdef ENABLE_IB
using communicator_t = ib_communicator_t;
#else
using communicator_t = mpi_communicator_t;
#endif

using communicator_ptr_t = std::shared_ptr<communicator_t>;

}

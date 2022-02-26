#pragma once

#include "../server/static_config.h"
#include "memory_storage.h"
#include "nvme_storage.h"
#include <cstddef>
#include <memory>

namespace bbts {

using storage_t = std::conditional<static_config::enable_storage, nvme_storage_t, memory_storage_t>::type;
using storage_ptr_t = std::shared_ptr<storage_t>;

template <class T = storage_t>
std::vector<std::thread> create_storage_threads(size_t num_threads, T &storage) {

  std::vector<std::thread> storage_req_threads;
  if constexpr (static_config::enable_storage) {

    // make the threads
    storage_req_threads.reserve(num_threads);
    for (node_id_t t = 0; t < num_threads; ++t) {
      storage_req_threads.push_back(
          std::thread([&]() { storage.request_thread(); }));
    }
  }

  return std::move(storage_req_threads);
}

} // namespace bbts
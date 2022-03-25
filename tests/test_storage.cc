#include <cstdlib>
#include <gtest/gtest.h>
#include <thread>
#include "../src/storage/storage.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/tensor/builtin_formats.h"

namespace bbts {

TEST(TestStorage, TestTwoTransactionSingleThreaded) {

  //
  storage_ptr_t storage;
  if constexpr(static_config::enable_storage) {
   storage = std::make_shared<storage_t>(nullptr, 1024 * 1024, "tmp.sto");
  }
  else {
    storage = std::make_shared<storage_t>(nullptr);
  }

  tensor_factory_ptr_t tf = std::make_shared<tensor_factory_t>();

  auto fmt_id = tf->get_tensor_ftm("dense");

  // make the meta
  dense_tensor_meta_t dm{fmt_id, 100, 100};

  // get the size of the tensor we need to crate
  auto tensor_size = tf->get_tensor_size(dm);

  // kick of a single request thread
  std::thread t = std::thread([&]() {
    storage->request_thread();
  });

  // run the local transaction to create the tensor
  storage->local_transaction({}, {{0, tensor_size}}, [&](const storage_t::reservation_result_t &res) {

    // create the tensor
    auto ts = res.create_or_get[0].get().tensor;

    // init the tensor
    auto &dt = tf->init_tensor(ts, dm).as<dense_tensor_t>();

    // write some memory into it
    for(int j = 0; j < 100 * 100; j++) {
      dt.data()[j] = (float) j;
    }
  });

  // run the local transaction to
  storage->local_transaction({ (tid_t) 0 }, {}, [&](const storage_t::reservation_result_t &res) {

    // get the dense tensor
    auto &dt = res.get[0].get().tensor->as<dense_tensor_t>();;

    // write some memory into it
    for(int j = 0; j < 100 * 100; j++) {
      EXPECT_LE(std::abs(dt.data()[j]  - (float) j), 0.0001f);
    }
  });

  // shutdown the storage
  storage->shutdown();

  // wait for the thread to finish
  t.join();
}


TEST(TestStorage, TestNoEvictionMultiThreaded) {

  // how many threads are going to be hammering the storage
  const int32_t num_threads = 8;
  const int32_t num_matrices = 10;

  // the thraeds that are going to hammer it
  std::vector<std::thread> threads;

  // create the storage
  storage_ptr_t storage;
  if constexpr(static_config::enable_storage) {
    storage = std::make_shared<storage_t>(nullptr, 100 * 1024 * 1024, "tmp.sto");
  }
  else {
    storage = std::make_shared<storage_t>(nullptr);
  }

  // make a tensor factory
  tensor_factory_ptr_t tf = std::make_shared<tensor_factory_t>();

  // kick of a single request thread
  std::thread rt = std::thread([&]() {
    storage->request_thread();
  });

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // make the threads
  threads.reserve(num_threads);
  for(int t = 0; t < num_threads; ++t) {

    // run a bunch of threads
    threads.emplace_back([&storage, &tf, &num_matrices, fmt_id, t]() {

      for(uint32_t i = 0; i < num_matrices; i++) {

        // make the meta
        dense_tensor_meta_t dm{fmt_id, i * 100, i * 200};

        // get the size of the tensor we need to crate
        auto tensor_size = tf->get_tensor_size(dm);

        storage->local_transaction({}, {{i + t * num_matrices, tensor_size}}, [&](const storage_t::reservation_result_t &res) {

          // crate the tensor
          auto ts = res.create_or_get[0].get().tensor;

          // init the tensor
          auto &dt = tf->init_tensor(ts, dm).as<dense_tensor_t>();

          // write some memory into it
          for(int j = 0; j < (i * 100) * (i * 200); j++) {
            dt.data()[j] = (float) j;
          }
        });
      }

      // get the tensors and check them
      for(size_t i = 0; i < num_matrices; i++) {

        storage->local_transaction({ (tid_t) (i + t * num_matrices) }, {}, [&](const storage_t::reservation_result_t &res) {

          // get the dense tensor
          auto &dt = res.get[0].get().tensor->as<dense_tensor_t>();;

          // write some memory into it
          for(int j = 0; j < (i * 100) * (i * 200); j++) {
            EXPECT_LE(std::abs(dt.data()[j]  - (float) j), 0.0001f);
          }

        });

        // remove the tensor
        storage->remove_by_tid((tid_t) (i + t * num_matrices));
      }
    });
  }

  // wait for all the threads to finish
  for(auto &t : threads) {
    t.join();
  }

    // shutdown the storage
  storage->shutdown();

  // wait for the thread to finish
  rt.join();
}


TEST(TestStorage, TestEvictionMultiThreaded) {

  // how many threads are going to be hammering the storage
  const int32_t num_threads = 10;
  const int32_t num_matrices = 40;

  // the thraeds that are going to hammer it
  std::vector<std::thread> threads;

  // create the storage
  storage_ptr_t storage;
  if constexpr(static_config::enable_storage) {
   storage = std::make_shared<storage_t>(nullptr, 1024 * 1024, "tmp.sto");
  }
  else {
    storage = std::make_shared<storage_t>(nullptr);
  }

  // make a tensor factory
  tensor_factory_ptr_t tf = std::make_shared<tensor_factory_t>();

  // kick of a single request thread
  std::thread rt = std::thread([&]() {
    storage->request_thread();
  });

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // make the threads
  threads.reserve(num_threads);
  for(int t = 0; t < num_threads; ++t) {

    // run a bunch of threads
    threads.emplace_back([&storage, &tf, &num_matrices, fmt_id, t]() {

      for(uint32_t i = 0; i < num_matrices; i++) {

        // make the meta
        dense_tensor_meta_t dm{fmt_id, 100, 100};

        // get the size of the tensor we need to crate
        auto tensor_size = tf->get_tensor_size(dm);

        storage->local_transaction({}, {{i + t * num_matrices, tensor_size}}, [&](const storage_t::reservation_result_t &res) {

          // crate the tensor
          auto ts = res.create_or_get[0].get().tensor;

          // init the tensor
          auto &dt = tf->init_tensor(ts, dm).as<dense_tensor_t>();

          // write some memory into it
          for(int j = 0; j < 100 * 100; j++) {
            dt.data()[j] = (float) (j + i);
          }
        });
      }

      // get the tensors and check them
      for(size_t i = 0; i < num_matrices; i++) {

        storage->local_transaction({ (tid_t) (i + t * num_matrices) }, {}, [&](const storage_t::reservation_result_t &res) {

          // get the dense tensor
          auto &dt = res.get[0].get().tensor->as<dense_tensor_t>();;

          // write some memory into it
          for(int j = 0; j < 100 * 100; j++) {
            EXPECT_LE(std::abs(dt.data()[j]  - (float) (j + i)), 0.0001f);
          }

        });

        // remove the tensor
        storage->remove_by_tid((tid_t) (i + t * num_matrices));
      }
    });
  }

  // wait for all the threads to finish
  for(auto &t : threads) {
    t.join();
  }

    // shutdown the storage
  storage->shutdown();

  // wait for the thread to finish
  rt.join();
}


TEST(TestStorage, TestEvictionMultiThreadedMultiRequestThreads1) {

  // how many threads are going to be hammering the storage
  const int32_t num_threads = 60;
  const int32_t num_matrices = 40;
  const int32_t num_req_threads = 50;

  // the thraeds that are going to hammer it
  std::vector<std::thread> threads;

  // create the storage
  storage_ptr_t storage;
  if constexpr(static_config::enable_storage) {
   storage = std::make_shared<storage_t>(nullptr, 1024 * 1024, "tmp.sto");
  }
  else {
    storage = std::make_shared<storage_t>(nullptr);
  }

  // make a tensor factory
  tensor_factory_ptr_t tf = std::make_shared<tensor_factory_t>();

  // kick of a single request thread
  std::vector<std::thread> rts; rts.reserve(num_req_threads);
  for(auto i = 0; i < num_req_threads; ++i) {
      rts.push_back(std::move(std::thread([&]() {
      storage->request_thread();
    })));
  }

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // make the threads
  threads.reserve(num_threads);
  for(int t = 0; t < num_threads; ++t) {

    // run a bunch of threads
    threads.emplace_back([&storage, &tf, &num_matrices, fmt_id, t]() {

      for(uint32_t i = 0; i < num_matrices; i++) {

        // make the meta
        dense_tensor_meta_t dm{fmt_id, 100, 100};

        // get the size of the tensor we need to crate
        auto tensor_size = tf->get_tensor_size(dm);

        storage->local_transaction({}, {{i + t * num_matrices, tensor_size}}, [&](const storage_t::reservation_result_t &res) {

          // crate the tensor
          auto ts = res.create_or_get[0].get().tensor;

          // init the tensor
          auto &dt = tf->init_tensor(ts, dm).as<dense_tensor_t>();

          // write some memory into it
          for(int j = 0; j < 100 * 100; j++) {
            dt.data()[j] = (float) (j + i);
          }
        });
      }

      // get the tensors and check them
      for(size_t i = 0; i < num_matrices; i++) {

        storage->local_transaction({ (tid_t) (i + t * num_matrices) }, {}, [&](const storage_t::reservation_result_t &res) {

          // get the dense tensor
          auto &dt = res.get[0].get().tensor->as<dense_tensor_t>();;

          // write some memory into it
          for(int j = 0; j < 100 * 100; j++) {
            EXPECT_LE(std::abs(dt.data()[j]  - (float) (j + i)), 0.0001f);
          }

        });

        // remove the tensor
        storage->remove_by_tid((tid_t) (i + t * num_matrices));
      }
    });
  }

  // wait for all the threads to finish
  for(auto &t : threads) {
    t.join();
  }

    // shutdown the storage
  storage->shutdown();

  // wait for the thread to finish
  for(auto &rt : rts) {
    rt.join();
  }

}

TEST(TestStorage, TestEvictionMultiThreadedMultiRequestThreads2) {

  // how many threads are going to be hammering the storage
  const int32_t num_threads = 10;
  const int32_t num_matrices = 300;
  const int32_t num_req_threads = 10;

  // the thraeds that are going to hammer it
  std::vector<std::thread> threads;

  // create the storage
  storage_ptr_t storage;
  if constexpr(static_config::enable_storage) {
   storage = std::make_shared<storage_t>(nullptr, 1024 * 1024, "tmp.sto");
  }
  else {
    storage = std::make_shared<storage_t>(nullptr);
  }

  // make a tensor factory
  tensor_factory_ptr_t tf = std::make_shared<tensor_factory_t>();

  // kick of a single request thread
  std::vector<std::thread> rts; rts.reserve(num_req_threads);
  for(auto i = 0; i < num_req_threads; ++i) {
      rts.push_back(std::move(std::thread([&]() {
      storage->request_thread();
    })));
  }

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // make the threads
  threads.reserve(num_threads);
  for(int t = 0; t < num_threads; ++t) {

    // run a bunch of threads
    threads.emplace_back([&storage, &tf, &num_matrices, fmt_id, t]() {

      for(uint32_t i = 0; i < num_matrices; i += 3) {

        // make the meta
        dense_tensor_meta_t dm{fmt_id, 100, 100};

        // get the size of the tensor we need to crate
        auto tensor_size = tf->get_tensor_size(dm);

        // std::cout << i +       t * num_matrices * 3 << '\n' << std::flush;
        // std::cout << i +  1 +  t * num_matrices * 3 << '\n' << std::flush;
        // std::cout << i +  2 +  t * num_matrices * 3 << '\n' << std::flush;
        storage->local_transaction({}, { {i +     t * num_matrices * 3, tensor_size},
                                         {i + 1 + t * num_matrices * 3, tensor_size},
                                         {i + 2 + t * num_matrices * 3, tensor_size} }, [&](const storage_t::reservation_result_t &res) {

          // crate the tensor
          auto ts0 = res.create_or_get[0].get().tensor;
          auto ts1 = res.create_or_get[1].get().tensor;
          auto ts2 = res.create_or_get[2].get().tensor;

          // init the tensor
          auto &dt0 = tf->init_tensor(ts0, dm).as<dense_tensor_t>();
          auto &dt1 = tf->init_tensor(ts1, dm).as<dense_tensor_t>();
          auto &dt2 = tf->init_tensor(ts2, dm).as<dense_tensor_t>();

          // write some memory into it
          for(int j = 0; j < 100 * 100; j++) {
            dt0.data()[j] = (float) (j + i);
            dt1.data()[j] = (float) (j + i + 1);
            dt2.data()[j] = (float) (j + i + 2);
          }
        });
      }

      // get the tensors and check them
      for(size_t i = 0; i < num_matrices; i += 3) {

        // std::cout << i +       t * num_matrices * 3 << '\n' << std::flush;
        // std::cout << i +  1 +  t * num_matrices * 3 << '\n' << std::flush;
        // std::cout << i +  2 +  t * num_matrices * 3 << '\n' << std::flush;
        storage->local_transaction({ (tid_t) (i +     t * num_matrices * 3),
                                     (tid_t) (i + 1 + t * num_matrices * 3),
                                     (tid_t) (i + 2 + t * num_matrices * 3) }, {}, [&](const storage_t::reservation_result_t &res) {

          // get the dense tensor
          auto &dt0 = res.get[0].get().tensor->as<dense_tensor_t>();
          auto &dt1 = res.get[1].get().tensor->as<dense_tensor_t>();
          auto &dt2 = res.get[2].get().tensor->as<dense_tensor_t>();

          // write some memory into it
          for(int j = 0; j < 100 * 100; j++) {
            EXPECT_LE(std::abs(dt0.data()[j]  - (float) (j + i)), 0.0001f);
            EXPECT_LE(std::abs(dt1.data()[j]  - (float) (j + i + 1)), 0.0001f);
            EXPECT_LE(std::abs(dt2.data()[j]  - (float) (j + i + 2)), 0.0001f);
          }

        });

        // remove the tensor
        storage->remove_by_tid((tid_t) (i +     t * num_matrices * 3));
        storage->remove_by_tid((tid_t) (i + 1 + t * num_matrices * 3));
        storage->remove_by_tid((tid_t) (i + 2 + t * num_matrices * 3));
      }
    });
  }

  // wait for all the threads to finish
  for(auto &t : threads) {
    t.join();
  }

    // shutdown the storage
  storage->shutdown();

  // wait for the thread to finish
  for(auto &rt : rts) {
    rt.join();
  }

}


TEST(TestStorage, TestEvictionMultiThreadedMultiRead) {

  // how many threads are going to be hammering the storage
  const int32_t num_threads = 100;
  const int32_t num_matrices = 1000;
  const int32_t num_gets = 300;
  const int32_t num_req_threads = 10;

  // the thraeds that are going to hammer it
  std::vector<std::thread> threads;

  // create the storage
  storage_ptr_t storage;
  if constexpr(static_config::enable_storage) {
   storage = std::make_shared<storage_t>(nullptr, 1024 * 1024, "tmp.sto");
  }
  else {
    storage = std::make_shared<storage_t>(nullptr);
  }

  // make a tensor factory
  tensor_factory_ptr_t tf = std::make_shared<tensor_factory_t>();

  // kick of a single request thread
  std::vector<std::thread> rts; rts.reserve(num_req_threads);
  for(auto i = 0; i < num_req_threads; ++i) {
      rts.push_back(std::move(std::thread([&]() {
      storage->request_thread();
    })));
  }

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  for(uint32_t i = 0; i < num_matrices; i += 3) {

    // make the meta
    dense_tensor_meta_t dm{fmt_id, 100, 100};

    // get the size of the tensor we need to crate
    auto tensor_size = tf->get_tensor_size(dm);

    // create the tensors
    storage->local_transaction({}, { {i    , tensor_size},
                                     {i + 1, tensor_size},
                                     {i + 2, tensor_size} }, [&](const storage_t::reservation_result_t &res) {

      // crate the tensor
      auto ts0 = res.create_or_get[0].get().tensor;
      auto ts1 = res.create_or_get[1].get().tensor;
      auto ts2 = res.create_or_get[2].get().tensor;

      // init the tensor
      auto &dt0 = tf->init_tensor(ts0, dm).as<dense_tensor_t>();
      auto &dt1 = tf->init_tensor(ts1, dm).as<dense_tensor_t>();
      auto &dt2 = tf->init_tensor(ts2, dm).as<dense_tensor_t>();

      // write some memory into it
      for(int j = 0; j < 100 * 100; j++) {
        dt0.data()[j] = (float) (j + i);
        dt1.data()[j] = (float) (j + i + 1);
        dt2.data()[j] = (float) (j + i + 2);
      }
    });
  }

  // make the threads
  threads.reserve(num_threads);
  for(int t = 0; t < num_threads; ++t) {

    // run a bunch of threads
    threads.emplace_back([&storage, &tf, &num_matrices, fmt_id]() {

      // get the tensors and check them
      for(uint32_t i = 0; i < num_gets; i++) {

        // the tensors we are getting
        tid_t t0 = (i * 3 + 0) % num_matrices;
        tid_t t1 = (i * 3 + 1) % num_matrices;
        tid_t t2 = (i * 3 + 2) % num_matrices;

        // check the tensors
        storage->local_transaction({ t0, t1, t2 }, {}, [&](const storage_t::reservation_result_t &res) {

          // get the dense tensor
          auto &dt0 = res.get[0].get().tensor->as<dense_tensor_t>();
          auto &dt1 = res.get[1].get().tensor->as<dense_tensor_t>();
          auto &dt2 = res.get[2].get().tensor->as<dense_tensor_t>();

          // write some memory into it
          for(int j = 0; j < 100 * 100; j++) {
            EXPECT_LE(std::abs(dt0.data()[j]  - (float) (j + t0)), 0.0001f);
            EXPECT_LE(std::abs(dt1.data()[j]  - (float) (j + t1)), 0.0001f);
            EXPECT_LE(std::abs(dt2.data()[j]  - (float) (j + t2)), 0.0001f);
          }

        });
      }
    });
  }

  // wait for all the threads to finish
  for(auto &t : threads) {
    t.join();
  }

  // shutdown the storage
  storage->shutdown();

  // remove the tensors
  for(uint32_t i = 0; i < num_matrices; i += 3) {

    storage->remove_by_tid((tid_t) (i));
    storage->remove_by_tid((tid_t) (i + 1));
    storage->remove_by_tid((tid_t) (i + 2));
  }

  // wait for the thread to finish
  for(auto &rt : rts) {
    rt.join();
  }
}

}

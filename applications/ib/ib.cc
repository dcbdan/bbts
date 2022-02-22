#include "../../src/communication/infiniband/connection.h"

#include <iostream>
#include <fstream>

// std::this_thread::sleep_for(std::chrono::seconds(1));
#include <chrono>
#include <thread>

using namespace bbts::ib;

//template <typename GenConnection>
//void three_node(int rank, size_t size, size_t num, GenConnection g) {
//  std::vector<bytes_t> bs;
//  for(int i = 0; i != 2*num; ++i) {
//    float* data = new float[size];
//    if(i < num && rank == 0) {
//      std::fill(data, data + size, 1000 + i);
//    }
//    if(i >= num && rank == 1) {
//      std::fill(data, data + size, 1000 + i);
//    }
//    bs.push_back({data, size});
//  }
//
//  std::vector<std::future<bool>> futs;
//
//  connection_t c = g();
//  std::cout << " CONNECTED " << std::endl;
//
//  if(rank == 0) {
//    for(int i = 0; i != num; ++i) {
//      futs.push_back(c.send_bytes_wait(1, 100+i, bs[i]));
//      futs.push_back(c.send_bytes_wait(2, 100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//      futs[idx-2].wait();
//    }
//    for(int i = num; i != 2*num; ++i) {
//      futs.push_back(c.recv_bytes_wait(100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//    }
//  } else if(rank == 1) {
//    for(int i = 0; i != num; ++i) {
//      futs.push_back(c.recv_bytes_wait(100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//    }
//    for(int i = num; i != 2*num; ++i) {
//      futs.push_back(c.send_bytes_wait(0, 100+i, bs[i]));
//      futs.push_back(c.send_bytes_wait(2, 100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//      futs[idx-2].wait();
//    }
//  } else if(rank == 2) {
//    // everything sends here
//    for(int i = 0; i != 2*num; ++i) {
//      futs.push_back(c.recv_bytes_wait(100 + i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//    }
//  }
//
//  for(auto& fut: futs) {
//    fut.get();
//  }
//
//  //for(bytes_t b: bs) {
//  //  float* data = (float*)b.data;
//  //  std::cout << " @ " << data[0];
//  //  delete[] data;
//  //}
//  //std::cout << std::endl;
//
//}

template <typename GenConnection>
void two_node(int rank, size_t size, size_t num, GenConnection g) {
  std::vector<bytes_t> bs;
  for(int i = 0; i != num; ++i) {
    float* data = new float[size];
    std::fill(data, data + size, 1000 + i);
    bs.push_back({data, sizeof(float)*size});
  }

  connection_t c = g();
  std::cout << " CONNECTED " << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::future<bool>> sends;
  std::vector<std::future<std::tuple<bool, own_bytes_t>>> recvs;
  if(rank == 0) {
    // send tag [100, 100+num)
    // recv tag [100+num,100+2*num)
    for(int i = 0; i != num; ++i) {
      sends.push_back(c.send(1, 100 + i, bs[i]));
      recvs.push_back(c.recv_from(1, 100 + num + i));
      //sends.back().wait();
      //recvs.back().wait();
    }
  } else if(rank == 1) {
    // send tag [100, 100+num)
    // recv tag [100+num,100+2*num)
    for(int i = 0; i != num; ++i) {
      sends.push_back(c.send(0, 100 +num + i, bs[i]));
      recvs.push_back(c.recv_from(0, 100 + i));
      //sends.back().wait();
      //recvs.back().wait();
    }
  }

  int num_success = 0;
  for(auto& fut: recvs) {
    auto [success, r] = fut.get();
    num_success += success;
    bs.push_back({
      .data = r.ptr.release(),
      .size = r.size
    });
  }
  for(auto& fut: sends) {
    num_success += fut.get();
  }

  std::cout << num_success << " / " << (2*num) << std::endl;
  auto stop = std::chrono::high_resolution_clock::now();
  // in bytes
  size_t total = size*num_success*sizeof(float);
  // in microseconds
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  if(rank < 2) {
    double gigabyte_per_second = (1.0e6 * total) / (duration.count() * 1.0e9);
    std::cout << std::endl << gigabyte_per_second << " GB/s" << std::endl;
  }

  for(bytes_t b: bs) {
    float* data = (float*)b.data;
    //if(rank < 2) {
    //  std::cout << " @ " << data[0];
    //}
    delete[] data;
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  std::string usage = "usage: " + std::string(argv[0]) + " <rank> <device name> <hosts file>";
  if(argc != 4) {
    std::cout << usage << std::endl;
    return 1;
  }

  int rank = std::stoi(argv[1]);

  std::vector<std::string> ips;
  {
    std::ifstream s = std::ifstream(argv[3]);
    if(!s.is_open()) {
      std::cout << "Could not open '" << argv[3] << "'" << std::endl;
      std::cout << usage << std::endl;
      return 1;
    }
    std::string l;
    while(std::getline(s, l)) {
      ips.push_back(l);
    }
    for(auto i: ips){
      std::cout << "IP: " << i << std::endl;
    }
  }

  size_t size = 1024;
  size_t num = 1000;

  //size_t size = 1024; //10240*10240*4;
  //size_t num = 1000;

  //three_node(rank, size, num, [&](){ return connection_t(argv[2], rank, ips); });
  two_node(rank, size, num, [&](){ return connection_t(argv[2], rank, ips); });

}

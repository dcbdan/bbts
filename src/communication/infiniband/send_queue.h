#pragma once

#include "connection.h"
#include "mr_bytes.h"

namespace bbts {
namespace ib {

struct send_item_t {
  memory_region_bytes_t bytes;
  std::promise<bool> pr;
};

struct virtual_send_queue_t {
  virtual_send_queue_t(connection_t* connection, int32_t rank, tag_t tag):
    which_state(state::wait), connection(connection), rank(rank), tag(tag)
  {}

  void insert_item(send_item_t && item);

  void recv_open_recv(uint64_t addr, uint64_t size, uint32_t key);
  void recv_fail_recv();

  void completed_open_send();
  void completed_rdma_write();
  void completed_close_send();
  void completed_fail_send();

  bool empty() const;
  size_t size() const;
private:
  //  -- waiting to post open send
  //  -- an open send has posted
  //  -- an open send has completed, waiting for open recv
  //  -- got open recv, an rdma has posted
  //  -- an rdma has completed, a send close has posted
  //  -- a send close has completed
  enum state {
    wait,
    post_send,
    post_rdma,
    post_close,
    post_fail
  } which_state;

  void process_next();
  send_item_t* get_head(state correct_state);
  void check_state(state correct_state) const;
  void post_open_send();

  // Invariant: only the front item is in process at a time
  std::queue<send_item_t> items;

  connection_t* connection;
  int32_t rank;
  tag_t tag;
};

} // namespace ib
} // namespace bbts


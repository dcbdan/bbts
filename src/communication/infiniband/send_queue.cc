#include "send_queue.h"

namespace bbts {
namespace ib {

void virtual_send_queue_t::insert_item(send_item_ptr_t item) {
  items.push(std::move(item));
  if(items.size() == 1) {
    post_open_send();
  }
}

void virtual_send_queue_t::completed_open_send() {
  // NOTE: It is possible to recv_open_recv or recv_fail_recv
  //       before completed_open_send, so there is no use having
  //       a transition from post_send to fini_send.
}

void virtual_send_queue_t::recv_open_recv(uint64_t addr, uint64_t size, uint32_t key) {
  send_item_ptr_t item = get_head(state::post_send);

  // allocate memory region if necessary... if that doesn't work,
  // tell the remote recv that the send isn't gonna happen
  bool success = item->bytes.setup_mr(connection, IBV_ACCESS_LOCAL_WRITE);

  if(!success) {
    connection->post_fail_send(rank, tag);
    which_state = state::post_fail;
    return;
  }

  // TODO: what happens when size of write > 2^31?
  connection->post_rdma_write(rank,
  {
    .wr_id = tag,
    .local_addr = (void*)(item->bytes.get_addr()),
    .local_size = item->bytes.get_size(), // TODO: narrowing conversion
    .local_key = item->bytes.get_local_key(),
    .remote_addr = addr,
    .remote_key = key
  });
  which_state = state::post_rdma;
}

void virtual_send_queue_t::recv_fail_recv() {
  send_item_ptr_t item = get_head(state::post_send);
  item->pr.set_value(false);
  items.pop();
  which_state = state::wait;
  process_next();
}

void virtual_send_queue_t::completed_rdma_write() {
  check_state(state::post_rdma);
  connection->post_close_send(rank, tag);
  which_state = state::post_close;
}

void virtual_send_queue_t::completed_close_send() {
  send_item_ptr_t item = get_head(state::post_close);
  item->pr.set_value(true);
  items.pop();
  which_state = state::wait;
  process_next();
}

void virtual_send_queue_t::completed_fail_send() {
  send_item_ptr_t item = get_head(state::post_fail);
  item->pr.set_value(false);
  items.pop();
  which_state = state::wait;
  process_next();
}

bool virtual_send_queue_t::empty() const {
  return items.empty();
}

size_t virtual_send_queue_t::size() const {
  return items.size();
}

void virtual_send_queue_t::process_next() {
  if(!items.empty()) {
    post_open_send();
  }
}

send_item_ptr_t virtual_send_queue_t::get_head(state correct_state) {
  if(items.empty()) {
    throw std::runtime_error("empty virtual send queue");
  }

  check_state(correct_state);

  return items.front();
}

void virtual_send_queue_t::check_state(state correct_state) const {
  if(which_state != correct_state) {
    _DCB_COUT_("expected state " << correct_state << " | which state " << which_state << std::endl);
    throw std::runtime_error("virtual send queue incorrect head state");
  }
}

void virtual_send_queue_t::post_open_send() {
  send_item_ptr_t item = get_head(state::wait);
  connection->post_open_send(rank, tag, item->bytes.get_size());
  which_state = state::post_send;
}

} // namespace ib
} // namespace bbts


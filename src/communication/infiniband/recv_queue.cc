#include "recv_queue.h"

namespace bbts {
namespace ib {

bool recv_item_t::acquire() {
  if(which_state != wait_open_send || pr.index() == which_pr::no_pr) {
    throw std::runtime_error("invalid acquire");
  }

  if(pr.index() == which_pr::rank_bytes || pr.index() == which_pr::rank_success) {
    bool prev_value = _acquired.fetch_or(true);
    // if it wasn't previously acquired, we have got it now
    return !prev_value;
  } else {
    return true;
  }
}

bool recv_item_t::is_acquired() const {
  if(which_state != wait_open_send || pr.index() == which_pr::no_pr) {
    return false;
  }
  return _acquired;
}

void recv_item_t::set_fail(int32_t rank) {
  auto which = pr.index();
  if(which == which_pr::no_pr) {
    throw std::runtime_error("set_fail: no promise to set");
  } else if(which == which_pr::rank_bytes) {
    auto& _pr = std::get<which_pr::rank_bytes>(pr);
    _pr.set_value({false, rank, own_bytes_t()});
  } else if(which == which_pr::rank_success) {
    auto& _pr = std::get<which_pr::rank_success>(pr);
    _pr.set_value({false, rank});
  } else if(which == which_pr::just_bytes) {
    auto& _pr = std::get<which_pr::just_bytes>(pr);
    _pr.set_value({false, own_bytes_t()});
  } else if(which == which_pr::just_success) {
    auto& _pr = std::get<which_pr::just_success>(pr);
    _pr.set_value(false);
  }
}

void recv_item_t::set_success(int32_t rank) {
  auto which = pr.index();
  if(which == which_pr::no_pr) {
    throw std::runtime_error("set_success: no promise to set");
  } else if(which == which_pr::rank_bytes) {
    auto& _pr = std::get<which_pr::rank_bytes>(pr);
    _pr.set_value({true, rank, bytes.extract_bytes()});
  } else if(which == which_pr::rank_success) {
    auto& _pr = std::get<which_pr::rank_success>(pr);
    _pr.set_value({true, rank});
  } else if(which == which_pr::just_bytes) {
    auto& _pr = std::get<which_pr::just_bytes>(pr);
    _pr.set_value({true, bytes.extract_bytes()});
  } else if(which == which_pr::just_success) {
    auto& _pr = std::get<which_pr::just_success>(pr);
    _pr.set_value(true);
  }
}

void virtual_recv_queue_t::insert_item(recv_item_ptr_t item) {
  // if the item is acquired, there is no need to deal with it
  if(item->is_acquired()) {
    return;
  }

  // are there pending open sends?
  if(pending_sizes.empty()) {
    // nope, we're still waiting for an open send
    waiting_open_send_items.push(item);
  } else {
    // note that if there are items in pending_sizes, waiting_open_send_items
    // must be empty
    if(!waiting_open_send_items.empty()) {
      throw std::runtime_error("insert item: virtual recv queue invalid state");
    }

    // at this point, we want to own item in it's entirety
    // so that no other queue can deal with it
    bool acquired = item->acquire();
    // if we don't own it, someone else owns it so that is ok
    if(!acquired) {
      return;
    }

    // this is the size used when posting recv
    item->sz = pending_sizes.front();
    pending_sizes.pop();

    // we're waiting to be in the front of the queue to post the recv
    item->which_state = recv_item_t::state::wait_post;

    in_process_items.push(item);
    if(in_process_items.size() == 1) {
      post_open_recv();
    }
  }
}

void virtual_recv_queue_t::recv_open_send(int64_t size) {
  // are there already pending sends / sizes? nothing to do but wait
  if(!pending_sizes.empty()) {
    pending_sizes.push(size);
    return;
  }

  // Are there any items waiting for an open send?
  // See if we can acquire one.
  recv_item_ptr_t item = nullptr;
  while(!waiting_open_send_items.empty()) {
    item = waiting_open_send_items.front();
    waiting_open_send_items.pop();
    if(item->acquire()) {
      break;
    }
  }

  if(item == nullptr) {
    // we'll deal with this when we have a recv
    pending_sizes.push(size);
    return;
  }

  if(item->which_state != recv_item_t::state::wait_open_send) {
    throw std::runtime_error("acquired recv item has incorrect state");
  }

  // tell the item we have an open send
  item->sz = size;
  item->which_state = recv_item_t::state::wait_post;
  in_process_items.push(item);
  if(in_process_items.size() == 1) {
    // item is at the head of queue, post recv
    post_open_recv();
  }
}

void virtual_recv_queue_t::completed_open_recv() {
  // NOTE: It is possible to recv_close_send or recv_fail_send
  //       before being notified that the open recv was completed.
  //       As a result, the fini_recv state is unused.
}

void virtual_recv_queue_t::recv_close_send() {
  recv_item_t* item = get_head(recv_item_t::state::post_recv);
  item->set_success(rank);
  in_process_items.pop();
  process_next();
}

void virtual_recv_queue_t::recv_fail_send() {
  recv_item_t* item = get_head(recv_item_t::state::post_recv);
  item->set_fail(rank);
  in_process_items.pop();
  process_next();
}

void virtual_recv_queue_t::completed_fail_recv() {
  recv_item_t* item = get_head(recv_item_t::state::post_fail);
  item->set_fail(rank);
  in_process_items.pop();
  process_next();
}

bool virtual_recv_queue_t::empty() const {
  return in_process_items.empty() &&
         waiting_open_send_items.empty() &&
         pending_sizes.empty();
}

void virtual_recv_queue_t::process_next() {
  if(!in_process_items.empty()) {
    post_open_recv();
  }
}

void virtual_recv_queue_t::post_open_recv() {
  recv_item_t* item = get_head(recv_item_t::state::wait_post);
  // bytes may contain
  //   nothing,
  //   memory,
  //   memory that is registerd.
  // either verify the memory it does have is big enough or if it doesn't
  // have memory, allocate it. And make sure the memory is registered.
  bool success = item->bytes.setup_bytes_and_mr(
    item->sz,
    connection,
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

  if(!success) {
    item->which_state = recv_item_t::state::post_fail;
    connection->post_fail_recv(rank, tag);
  } else {
    item->which_state = recv_item_t::state::post_recv;
    connection->post_open_recv(rank, tag,
      item->bytes.get_addr(),
      item->bytes.get_size(),
      item->bytes.get_remote_key());
  }
}

// INVARIANT: we own this recv_item_t and no one else will modify it
// (No item should end up at the head of our queue unless we have activaly acquired it)
recv_item_t* virtual_recv_queue_t::get_head(recv_item_t::state correct_state) {
  if(in_process_items.empty()) {
    throw std::runtime_error("verify head: empty queue");
  }
  recv_item_t* item = in_process_items.front().get();
  if(item->which_state != correct_state) {
    _DCB_COUT_("expected state " << correct_state << " | which state " << (item->which_state) << std::endl);

    throw std::runtime_error("verify head: invalid head state");
  }
  return item;
}


} // namespace ib
} // namespace bbts


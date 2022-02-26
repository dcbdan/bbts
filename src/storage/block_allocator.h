#pragma once

#include <cassert>
#include <map>
#include <memory>

namespace bbts {

class block_allocator_t {
public:
  typedef size_t offset_type;
  static const offset_type invalid_offset = static_cast<offset_type>(-1);

private:
  struct free_block_info_t;

  // type of the map that keeps memory blocks sorted by their offsets
  using free_blocks_by_offset_map_t = std::map<offset_type, free_block_info_t>;

  // type of the map that keeps memory blocks sorted by their sizes
  using free_blocks_by_size_t =
      std::multimap<offset_type, free_blocks_by_offset_map_t::iterator>;

  struct free_block_info_t {

    // block size (no reserved space for the size of the allocation)
    offset_type size;

    // iterator referencing this block in the multimap sorted by the block size
    free_blocks_by_size_t::iterator order_by_size_it;

    free_block_info_t(offset_type _size) : size(_size) {}
  };

public:
  block_allocator_t(offset_type max_size)
      : _max_size(max_size), _free_size(max_size) {
    // insert single maximum-size block
    add_new_block(0, _max_size);
  }

  ~block_allocator_t() = default;

  block_allocator_t(block_allocator_t &&rhs)
      : _free_blocks_by_offset(std::move(rhs._free_blocks_by_offset)),
        _free_blocks_by_size(std::move(rhs._free_blocks_by_size)),
        _max_size(rhs._max_size), _free_size(rhs._free_size) {
    // rhs.m_max_size = 0; - const
    rhs._free_size = 0;
  }

  block_allocator_t &operator=(block_allocator_t &&rhs) = delete;
  block_allocator_t(const block_allocator_t &) = delete;
  block_allocator_t &operator=(const block_allocator_t &) = delete;

  offset_type allocate(offset_type size) {
    assert(size != 0);
    if (_free_size < size)
      return invalid_offset;

    // get the first block that is large enough to encompass size bytes
    // lower_bound() returns an iterator pointing to the first element that
    // is not less (i.e. >= ) than key
    auto smallest_block_it_it = _free_blocks_by_size.lower_bound(size);
    if (smallest_block_it_it == _free_blocks_by_size.end())
      return invalid_offset;

    auto smallest_block_it = smallest_block_it_it->second;
    assert(size <= smallest_block_it->second.size);
    assert(smallest_block_it->second.size == smallest_block_it_it->first);

    //        smallest_block_it.offset
    //        |                                  |
    //        |<------smallest_block_it.size------>|
    //        |<------size------>|<---new_size--->|
    //        |                  |
    //        offset              new_offset
    //
    auto offset = smallest_block_it->first;
    auto new_offset = offset + size;
    auto new_size = smallest_block_it->second.size - size;
    assert(smallest_block_it_it == smallest_block_it->second.order_by_size_it);
    _free_blocks_by_size.erase(smallest_block_it_it);
    _free_blocks_by_offset.erase(smallest_block_it);
    if (new_size > 0) {
      add_new_block(new_offset, new_size);
    }

    _free_size -= size;

    return offset;
  }

  void free(offset_type offset, offset_type size) {
    assert(offset + size <= _max_size);

    // find the first element whose offset is greater than the specified offset.
    // upper_bound() returns an iterator pointing to the first element in the
    // container whose key is considered to go after k.
    auto next_block_it = _free_blocks_by_offset.upper_bound(offset);

    // block being deallocated must not overlap with the next block
    assert(next_block_it == _free_blocks_by_offset.end() ||
           offset + size <= next_block_it->first);
    auto prev_block_it = next_block_it;
    if (prev_block_it != _free_blocks_by_offset.begin()) {
      --prev_block_it;
      // block being deallocated must not overlap with the previous block
      assert(offset >= prev_block_it->first + prev_block_it->second.size);
    } else {
      prev_block_it = _free_blocks_by_offset.end();
    }

    offset_type new_size, new_offset;
    if (prev_block_it != _free_blocks_by_offset.end() &&
        offset == prev_block_it->first + prev_block_it->second.size) {
      //       prev_block.offset          offset
      //       |                          |
      //       |<-----prev_block.size----->|<------size-------->|
      //
      new_size = prev_block_it->second.size + size;
      new_offset = prev_block_it->first;

      if (next_block_it != _free_blocks_by_offset.end() &&
          offset + size == next_block_it->first) {
        //     prev_block.offset          offset               next_block.offset
        //     |                          |                    |
        //     |<-----prev_block.size----->|<------size-------->|<-----next_block.size----->|
        //
        new_size += next_block_it->second.size;
        _free_blocks_by_size.erase(prev_block_it->second.order_by_size_it);
        _free_blocks_by_size.erase(next_block_it->second.order_by_size_it);
        // Delete the range of two blocks
        ++next_block_it;
        _free_blocks_by_offset.erase(prev_block_it, next_block_it);
      } else {
        //     prev_block.offset          offset next_block.offset
        //     |                          |                             |
        //     |<-----prev_block.size----->|<------size-------->| ~ ~ ~
        //     |<-----next_block.size----->|
        //
        _free_blocks_by_size.erase(prev_block_it->second.order_by_size_it);
        _free_blocks_by_offset.erase(prev_block_it);
      }
    } else if (next_block_it != _free_blocks_by_offset.end() &&
               offset + size == next_block_it->first) {
      //     prev_block.offset                  offset next_block.offset | | |
      //     |<-----prev_block.size----->| ~ ~ ~
      //     |<------size-------->|<-----next_block.size----->|
      //
      new_size = size + next_block_it->second.size;
      new_offset = offset;
      _free_blocks_by_size.erase(next_block_it->second.order_by_size_it);
      _free_blocks_by_offset.erase(next_block_it);
    } else {
      //     prev_block.offset                  offset next_block.offset
      //     |                                  |                            |
      //     |<-----prev_block.size----->| ~ ~ ~ |<------size-------->| ~ ~ ~
      //     |<-----next_block.size----->|
      //
      new_size = size;
      new_offset = offset;
    }

    add_new_block(new_offset, new_size);

    _free_size += size;
  }

  offset_type get_max_size() const { return _max_size; }
  bool is_full() const { return _free_size == 0; };
  bool is_empty() const { return _free_size == _max_size; };
  offset_type get_free_size() const { return _free_size; }

private:
  void add_new_block(offset_type offset, offset_type size) {
    auto new_block_it = _free_blocks_by_offset.emplace(offset, size);
    assert(new_block_it.second);
    auto order_it = _free_blocks_by_size.emplace(size, new_block_it.first);
    new_block_it.first->second.order_by_size_it = order_it;
  }

  free_blocks_by_offset_map_t _free_blocks_by_offset;
  free_blocks_by_size_t _free_blocks_by_size;

  const offset_type _max_size = 0;
  offset_type _free_size = 0;
};

using block_allocator_ptr_t = std::shared_ptr<block_allocator_t>;

} // namespace bbts

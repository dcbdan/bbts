#include "expand.h"

std::ostream& operator<<(std::ostream& out, utils::expand::part_dim_t const& p) {
  //out << "start, interval, full: " << p.start << " " << p.interval << " " << p.full;
  out << "[0, " << p.full << ") ["
    << p.start << "," << p.start + p.interval << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, utils::expand::expand_dim_t const& e) {
  using namespace utils::expand;

  out << "{in " <<
    (part_dim_t{ .start = e.start_inn, .interval = e.interval, .full = e.full_inn })
    << "out " <<
    (part_dim_t{ .start = e.start_out, .interval = e.interval, .full = e.full_out })
    << "}";

  return out;
}


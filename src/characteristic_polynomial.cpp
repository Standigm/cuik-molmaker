// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "characteristic_polynomial.h"

#include <algorithm>
#include <array>
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cuik_molmaker {

namespace {

using u64    = std::uint64_t;
using u128   = unsigned __int128;
using BigInt = boost::multiprecision::cpp_int;

//! The 64 largest primes below 2^62: residues modulo all of them pin down any integer of up
//! to about 3,960 bits, far past where Ipc leaves the double range.
constexpr std::array<u64, 64> PRIMES = {
  0x3fffffffffffffc7ull, 0x3fffffffffffffa9ull, 0x3fffffffffffff8bull, 0x3fffffffffffff71ull, 0x3fffffffffffff67ull,
  0x3fffffffffffff59ull, 0x3fffffffffffff55ull, 0x3fffffffffffff3dull, 0x3fffffffffffff35ull, 0x3ffffffffffffeefull,
  0x3ffffffffffffee1ull, 0x3ffffffffffffec3ull, 0x3ffffffffffffe45ull, 0x3ffffffffffffe1dull, 0x3ffffffffffffe11ull,
  0x3ffffffffffffdc1ull, 0x3ffffffffffffdbbull, 0x3ffffffffffffda5ull, 0x3ffffffffffffd87ull, 0x3ffffffffffffd69ull,
  0x3ffffffffffffd03ull, 0x3ffffffffffffcfbull, 0x3ffffffffffffcf7ull, 0x3ffffffffffffce9ull, 0x3ffffffffffffcd3ull,
  0x3ffffffffffffcc1ull, 0x3ffffffffffffc65ull, 0x3ffffffffffffc2bull, 0x3ffffffffffffc1full, 0x3ffffffffffffc17ull,
  0x3ffffffffffffc11ull, 0x3ffffffffffffc07ull, 0x3ffffffffffffb53ull, 0x3ffffffffffffb27ull, 0x3ffffffffffffaf3ull,
  0x3ffffffffffffab7ull, 0x3ffffffffffffa67ull, 0x3ffffffffffffa15ull, 0x3ffffffffffff9efull, 0x3ffffffffffff9d9ull,
  0x3ffffffffffff9d3ull, 0x3ffffffffffff9c5ull, 0x3ffffffffffff9afull, 0x3ffffffffffff977ull, 0x3ffffffffffff95full,
  0x3ffffffffffff95bull, 0x3ffffffffffff959ull, 0x3ffffffffffff8e1ull, 0x3ffffffffffff8a7ull, 0x3ffffffffffff889ull,
  0x3ffffffffffff87dull, 0x3ffffffffffff805ull, 0x3ffffffffffff7e7ull, 0x3ffffffffffff7c9ull, 0x3ffffffffffff7a3ull,
  0x3ffffffffffff775ull, 0x3ffffffffffff757ull, 0x3ffffffffffff739ull, 0x3ffffffffffff713ull, 0x3ffffffffffff6d1ull,
  0x3ffffffffffff6c1ull, 0x3ffffffffffff6b9ull, 0x3ffffffffffff6a3ull, 0x3ffffffffffff68bull,
};

//! Arithmetic modulo an odd prime p < 2^62 in Montgomery form (R = 2^64).
class MontgomeryField {
 public:
  explicit MontgomeryField(u64 prime) : p_(prime) {
    u64 inverse = p_;  // Newton iteration for p^-1 mod 2^64; each step doubles the valid bits
    for (int i = 0; i < 6; ++i) {
      inverse *= 2 - p_ * inverse;
    }
    neg_inverse_ = ~inverse + 1;
    const u64 r  = static_cast<u64>((static_cast<u128>(1) << 64) % p_);
    r_squared_   = static_cast<u64>(static_cast<u128>(r) * r % p_);
    one_         = to(1);
  }

  u64 prime() const { return p_; }
  u64 one() const { return one_; }
  u64 to(u64 a) const { return mul(a % p_, r_squared_); }
  u64 from(u64 a) const { return reduce(a); }
  u64 mul(u64 a, u64 b) const { return reduce(static_cast<u128>(a) * b); }
  u64 add(u64 a, u64 b) const { return a + b >= p_ ? a + b - p_ : a + b; }
  u64 sub(u64 a, u64 b) const { return a >= b ? a - b : a + p_ - b; }
  u64 inverse(u64 a) const {  // Fermat: a^(p-2)
    u64 result = one_;
    for (u64 e = p_ - 2; e; e >>= 1, a = mul(a, a)) {
      if (e & 1) {
        result = mul(result, a);
      }
    }
    return result;
  }

 private:
  u64 reduce(u128 t) const {  // t < p * 2^64, so t + m p < 2^127 and the result is below 2p
    const u64 m = static_cast<u64>(t) * neg_inverse_;
    const u64 r = static_cast<u64>((t + static_cast<u128>(m) * p_) >> 64);
    return r >= p_ ? r - p_ : r;
  }

  u64 p_, neg_inverse_, r_squared_, one_;
};

//! A symmetric elimination order for xI - A and the entries each step touches.
//!
//! Entries 0..n-1 are the diagonal; the rest are the off-diagonal pairs the elimination
//! reaches, original edges and fill-in alike, shared by both triangles.
struct EliminationPlan {
  struct Step {
    size_t              pivot;       //!< diagonal entry of the vertex eliminated
    std::vector<size_t> to_later;    //!< entries (v, a) for the vertices a still to come
    std::vector<size_t> later_diag;  //!< diagonal entries of those a
    std::vector<size_t> pairs;       //!< entries (a, b), a before b, row-major
  };
  size_t              num_vertices = 0;
  size_t              num_entries  = 0;
  std::vector<size_t> edges;  //!< entries holding -1 initially
  std::vector<Step>   steps;
};

//! Minimum-degree order: on a near-tree it eliminates leaves first and fills in little.
EliminationPlan plan_elimination(const std::vector<std::vector<unsigned int>>& neighbours) {
  EliminationPlan plan;
  plan.num_vertices = neighbours.size();
  plan.num_entries  = plan.num_vertices;
  // Molecular degrees are small, so adjacency and entry lookups are short linear scans.
  std::vector<std::vector<std::pair<size_t, size_t>>> adjacency(plan.num_vertices);  // (vertex, entry)
  const auto                                          connect = [&](size_t a, size_t b) {
    for (const auto& [vertex, entry] : adjacency[a]) {
      if (vertex == b) {
        return entry;
      }
    }
    const size_t entry = plan.num_entries++;
    adjacency[a].emplace_back(b, entry);
    adjacency[b].emplace_back(a, entry);
    return entry;
  };
  for (size_t v = 0; v < plan.num_vertices; ++v) {
    for (const unsigned int u : neighbours[v]) {
      if (v < u) {
        plan.edges.push_back(connect(v, u));
      }
    }
  }

  std::vector<bool> eliminated(plan.num_vertices, false);
  for (size_t step = 0; step < plan.num_vertices; ++step) {
    size_t v = plan.num_vertices;
    for (size_t u = 0; u < plan.num_vertices; ++u) {
      if (!eliminated[u] && (v == plan.num_vertices || adjacency[u].size() < adjacency[v].size())) {
        v = u;
      }
    }
    EliminationPlan::Step                        current{v, {}, {}, {}};
    const std::vector<std::pair<size_t, size_t>> later = adjacency[v];
    for (const auto& [vertex, entry] : later) {
      current.to_later.push_back(entry);
      current.later_diag.push_back(vertex);
    }
    for (size_t i = 0; i < later.size(); ++i) {
      for (size_t j = i + 1; j < later.size(); ++j) {
        current.pairs.push_back(connect(later[i].first, later[j].first));
      }
    }
    for (const auto& [vertex, entry] : later) {
      auto& list = adjacency[vertex];
      list.erase(std::find_if(list.begin(), list.end(), [v](const auto& item) { return item.first == v; }));
    }
    adjacency[v].clear();
    eliminated[v] = true;
    plan.steps.push_back(std::move(current));
  }
  return plan;
}

//! det(x I - A) mod p at `count` consecutive points from `first`, by LDL^T elimination run on
//! all points at once, so each pivot vector shares one inversion. A pivot that vanishes mod p
//! makes that point unusable; it is flagged instead.
void evaluate_determinants(const EliminationPlan& plan,
                           const MontgomeryField& field,
                           u64                    first,
                           size_t                 count,
                           std::vector<u64>&      determinants,
                           std::vector<bool>&     usable) {
  std::vector<u64> values(plan.num_entries * count, 0), inverse(count), prefix(count), multipliers;
  for (size_t v = 0; v < plan.num_vertices; ++v) {
    for (size_t j = 0; j < count; ++j) {
      values[v * count + j] = field.to(first + j);
    }
  }
  const u64 minus_one = field.sub(0, field.one());
  for (const size_t edge : plan.edges) {
    std::fill_n(&values[edge * count], count, minus_one);
  }
  determinants.assign(count, field.one());
  usable.assign(count, true);

  for (const auto& step : plan.steps) {
    u64* const pivot = &values[step.pivot * count];
    for (size_t j = 0; j < count; ++j) {
      if (pivot[j] == 0) {
        usable[j] = false;
        pivot[j]  = field.one();  // keeps the batch inversion defined; the point is discarded
      }
    }
    prefix[0] = pivot[0];
    for (size_t j = 1; j < count; ++j) {
      prefix[j] = field.mul(prefix[j - 1], pivot[j]);
    }
    u64 running = field.inverse(prefix[count - 1]);
    for (size_t j = count; j-- > 1;) {
      inverse[j] = field.mul(running, prefix[j - 1]);
      running    = field.mul(running, pivot[j]);
    }
    inverse[0] = running;
    for (size_t j = 0; j < count; ++j) {
      determinants[j] = field.mul(determinants[j], pivot[j]);
    }

    // Schur complement: M[a][b] -= M[a][v] M[v][b] / M[v][v] over the later neighbours.
    const size_t later = step.to_later.size();
    multipliers.assign(later * count, 0);
    for (size_t a = 0; a < later; ++a) {
      const u64* const column   = &values[step.to_later[a] * count];
      u64* const       factor   = &multipliers[a * count];
      u64* const       diagonal = &values[step.later_diag[a] * count];
      for (size_t j = 0; j < count; ++j) {
        factor[j]   = field.mul(column[j], inverse[j]);
        diagonal[j] = field.sub(diagonal[j], field.mul(factor[j], column[j]));
      }
    }
    size_t pair = 0;
    for (size_t a = 0; a < later; ++a) {
      for (size_t b = a + 1; b < later; ++b, ++pair) {
        const u64* const factor = &multipliers[a * count];
        const u64* const column = &values[step.to_later[b] * count];
        u64* const       target = &values[step.pairs[pair] * count];
        for (size_t j = 0; j < count; ++j) {
          target[j] = field.sub(target[j], field.mul(factor[j], column[j]));
        }
      }
    }
  }
}

//! Coefficients a_0 .. a_n of det(x I - A) mod p (a_i multiplies x^i), interpolated from
//! values at points above the largest degree.
std::vector<u64> coefficients_mod(const EliminationPlan& plan, const MontgomeryField& field, u64 first_point) {
  const size_t n = plan.num_vertices;

  // n + 1 values fix a degree-n polynomial; evaluate in blocks to bound memory.
  constexpr size_t  MAX_BLOCK = 256;
  std::vector<u64>  xs, ys, determinants;
  std::vector<bool> usable;
  for (u64 next = first_point; xs.size() < n + 1;) {
    const size_t count = std::min<size_t>(MAX_BLOCK, n + 1 - xs.size() + 4);
    evaluate_determinants(plan, field, next, count, determinants, usable);
    for (size_t j = 0; j < count && xs.size() < n + 1; ++j) {
      if (usable[j]) {
        xs.push_back(next + j);
        ys.push_back(determinants[j]);
      }
    }
    next += count;
  }

  // Newton divided differences. The points are increasing integers, so every gap is a small
  // positive integer, and their inverses come from one table: inv(i) = -(p / i) inv(p mod i).
  const size_t     span = static_cast<size_t>(xs.back() - xs.front());
  std::vector<u64> small_inverse(span + 1, 0);
  const u64        p = field.prime();
  if (span >= 1) {
    small_inverse[1] = 1;
  }
  for (size_t i = 2; i <= span; ++i) {
    small_inverse[i] = p - static_cast<u64>(static_cast<u128>(p / i) * small_inverse[p % i] % p);
  }
  for (u64& value : small_inverse) {
    value = field.to(value);
  }
  for (size_t k = 1; k <= n; ++k) {
    for (size_t i = n; i >= k; --i) {
      ys[i] = field.mul(field.sub(ys[i], ys[i - 1]), small_inverse[xs[i] - xs[i - k]]);
    }
  }

  // From the Newton form: P = ys[n]; P = P (x - xs[i]) + ys[i] for i = n-1 .. 0.
  std::vector<u64> polynomial(n + 1, 0);
  polynomial[0] = ys[n];
  for (size_t i = n, degree = 0; i-- > 0; ++degree) {
    const u64 xi = field.to(xs[i]);
    for (size_t t = degree + 1; t > 0; --t) {
      polynomial[t] = field.sub(polynomial[t - 1], field.mul(xi, polynomial[t]));
    }
    polynomial[0] = field.add(field.sub(0, field.mul(xi, polynomial[0])), ys[i]);
  }
  for (u64& c : polynomial) {
    c = field.from(c);
  }
  return polynomial;
}

//! log2 of an upper bound on every |c_k|.
//!
//! |c_k| = |e_k(lambda)| <= e_k(|lambda|) <= C(n,k) (sum |lambda| / n)^k (Maclaurin), and by
//! Cauchy-Schwarz sum |lambda| <= sqrt(n tr A^2) = sqrt(2 m n).
double coefficient_bits_bound(size_t n, size_t twice_m) {
  const double half_log_density = 0.5 * std::log2(static_cast<double>(twice_m) / static_cast<double>(n));
  const double log2_n_factorial = std::lgamma(n + 1.0) / std::log(2.0);
  double       bound            = 0.0;
  for (size_t k = 0; k <= n; ++k) {
    const double log2_binomial = log2_n_factorial - (std::lgamma(k + 1.0) + std::lgamma(n - k + 1.0)) / std::log(2.0);
    bound                      = std::max(bound, log2_binomial + k * half_log_density);
  }
  return bound;
}

//! |value| * 2^-exponent rounded to the nearest double, ties to even. With exponent 0 this
//! is what Python's float(int) gives; otherwise only results too small to matter next to
//! the largest coefficient can be rounded twice, on entering the subnormal range.
double magnitude(BigInt value, int exponent) {
  if (value < 0) {
    value = -value;
  }
  if (value == 0) {
    return 0.0;
  }
  const unsigned int top = boost::multiprecision::msb(value);
  if (top < 64) {
    return std::ldexp(static_cast<double>(static_cast<u64>(value)), -exponent);
  }
  // Keep the leading 64 bits and fold every dropped bit into the lowest one: that bit lies
  // below double's rounding position, so the conversion still rounds exactly.
  const unsigned int shift = top - 63;
  auto               head  = static_cast<u64>(value >> shift);
  if ((value & ((BigInt(1) << shift) - 1)) != 0) {
    head |= 1u;
  }
  return std::ldexp(static_cast<double>(head), static_cast<int>(shift) - exponent);
}

}  // namespace

PolynomialMagnitudes characteristic_polynomial_magnitudes(const RDKit::ROMol& mol) {
  const size_t n = mol.getNumAtoms();
  if (n == 0) {
    return {};
  }
  std::vector<std::vector<unsigned int>> neighbours(n);
  size_t                                 twice_m = 0, max_degree = 0;
  for (const RDKit::Bond* bond : mol.bonds()) {
    neighbours[bond->getBeginAtomIdx()].push_back(bond->getEndAtomIdx());
    neighbours[bond->getEndAtomIdx()].push_back(bond->getBeginAtomIdx());
    twice_m += 2;
  }
  PolynomialMagnitudes result{std::vector<double>(n + 1, 0.0)};
  result.scaled[0] = 1.0;
  if (twice_m == 0) {
    return result;  // det(x I) = x^n
  }
  for (const auto& list : neighbours) {
    max_degree = std::max(max_degree, list.size());
  }

  // Symmetric residues need a modulus above twice the largest |c_k|; two bits of slack
  // cover the floating-point evaluation of the bound.
  const double needed_bits = coefficient_bits_bound(n, twice_m) + 3.0;
  size_t       num_primes  = 0;
  for (double covered = 0.0; covered < needed_bits; ++num_primes) {
    if (num_primes == PRIMES.size()) {
      throw std::range_error("Ipc's characteristic polynomial is too large to compute exactly");
    }
    covered += std::log2(static_cast<double>(PRIMES[num_primes]));
  }

  // Every eigenvalue of every induced subgraph lies in [-max_degree, max_degree], so above
  // that no pivot vanishes over the integers; modulo p one does only by rare accident.
  const EliminationPlan plan     = plan_elimination(neighbours);
  const u64             first    = max_degree + 1;
  std::vector<u64>      residues = coefficients_mod(plan, MontgomeryField(PRIMES[0]), first);

  if (num_primes == 1) {
    const u64 p = PRIMES[0];
    for (size_t k = 0; k <= n; ++k) {
      const u64 r      = residues[n - k];  // c_k multiplies x^(n-k)
      result.scaled[k] = static_cast<double>(r > p / 2 ? p - r : r);
    }
    return result;
  }

  // Chinese remainder theorem, one prime at a time: x += M ((r - x) M^-1 mod p).
  std::vector<BigInt> exact(residues.begin(), residues.end());
  BigInt              modulus = PRIMES[0];
  for (size_t i = 1; i < num_primes; ++i) {
    const u64              p = PRIMES[i];
    const MontgomeryField  field(p);
    const std::vector<u64> r     = coefficients_mod(plan, field, first);
    const u64              inv_m = field.from(field.inverse(field.to(static_cast<u64>(modulus % p))));
    for (size_t k = 0; k <= n; ++k) {
      const u64 current = static_cast<u64>(exact[k] % p);
      const u64 delta   = r[k] >= current ? r[k] - current : r[k] + p - current;
      exact[k] += modulus * static_cast<u64>(static_cast<u128>(delta) * inv_m % p);
    }
    modulus *= p;
  }
  const BigInt half = modulus / 2;
  unsigned int top  = 0;
  for (BigInt& x : exact) {
    if (x > half) {
      x = modulus - x;  // the magnitude of the symmetric residue
    }
    if (x != 0) {
      top = std::max(top, static_cast<unsigned int>(boost::multiprecision::msb(x)));
    }
  }
  // The n + 1 magnitudes sum to less than 2^(top + 1 + bit_width(n + 1)); keep that a factor
  // of two inside the double range so that rounding cannot overflow either.
  const int sum_bits = static_cast<int>(top) + 2 + static_cast<int>(std::bit_width(n + 1));
  result.exponent    = std::max(0, sum_bits - std::numeric_limits<double>::max_exponent);
  for (size_t k = 0; k <= n; ++k) {
    result.scaled[k] = magnitude(exact[n - k], result.exponent);
  }
  return result;
}

}  // namespace cuik_molmaker

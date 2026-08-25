// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file Implements whole-molecule descriptors missing from RDKit's C++ library, and the
//!       threaded batch entry point declared in molecular_descriptors.h.

#include "molecular_descriptors.h"

#include "features.h"
#include "qed_data.h"

// C++ standard library headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// RDKit headers
#include <DataStructs/SparseIntVect.h>
#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/Descriptors/ConnectivityDescriptors.h>
#include <GraphMol/Descriptors/Crippen.h>
#include <GraphMol/Descriptors/Lipinski.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/Descriptors/MolSurf.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolPickler.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <RDGeneral/types.h>

namespace {

//! Information entropy in bits of a distribution given by unnormalised counts.
//! Mirrors `rdkit.ML.InfoTheory.entropy.InfoEntropy`.
double info_entropy(const std::vector<double>& counts) {
  double total = 0.0;
  for (const double count : counts) {
    total += count;
  }
  if (total <= 0.0) {
    return 0.0;
  }

  double entropy = 0.0;
  for (const double count : counts) {
    if (count > 0.0) {
      const double probability = count / total;
      entropy -= probability * std::log2(probability);
    }
  }
  return entropy;
}

//! Bond order as BertzCT counts it: aromatic bonds are 1.5, everything else is the
//! numeric value of its bond type. Matches `_LookUpBondOrder`.
double bertz_bond_order(const RDKit::Bond& bond) {
  if (bond.getIsAromatic() || bond.getBondType() == RDKit::Bond::AROMATIC) {
    return 1.5;
  }
  return static_cast<double>(bond.getBondType());
}

//! Rounds `value * 10^4` to the nearest integer, ties to even, from the exact value.
//!
//! This is what "%.4f" yields, without the decimal conversion. Doing it in floating
//! point instead -- `llround(value * 1e4)` -- disagrees at exact ties, which round the
//! other way: 0.40625 renders as "0.4062" but llrounds to 4063.
//!
//! Exact for `|value| < 2^49`, which every distance in a molecular graph satisfies;
//! larger magnitudes saturate rather than wrap, keeping the comparison total.
std::int64_t scaled_to_four_decimals(double value) {
  if (!std::isfinite(value) || value == 0.0) {
    return 0;
  }
  const bool   negative  = value < 0.0;
  const double magnitude = negative ? -value : value;

  int                 exponent = 0;
  const double        fraction = std::frexp(magnitude, &exponent);  // magnitude = fraction * 2^exponent
  const std::uint64_t mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, 53));

  // 10^4 is 2^4 * 625. Scaling by the odd factor alone keeps the product inside 64 bits
  // (mantissa < 2^53, so mantissa * 625 < 2^63) and the 2^4 folds into the shift, which
  // avoids the 128-bit integers MSVC does not provide.
  const std::uint64_t scaled = mantissa * 625u;
  const int           shift  = 49 - exponent;  // magnitude * 10^4 == scaled * 2^-shift

  std::uint64_t rounded = 0;
  if (shift >= 64) {
    // scaled < 2^63 <= 2^(shift-1), so the value lies below a half and rounds to zero.
    rounded = 0;
  } else if (shift > 0) {
    const std::uint64_t quotient  = scaled >> shift;
    const std::uint64_t remainder = scaled - (quotient << shift);
    const std::uint64_t half      = std::uint64_t{1} << (shift - 1);
    rounded = quotient + ((remainder > half || (remainder == half && (quotient & 1) != 0)) ? 1 : 0);
  } else {
    const int           left    = -shift;
    const std::uint64_t ceiling = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    rounded                     = (left < 64 && scaled <= (ceiling >> left)) ? (scaled << left) : ceiling;
  }

  const std::int64_t result = static_cast<std::int64_t>(rounded);
  return negative ? -result : result;
}

//! Groups atoms into symmetry classes by their sorted bond-order distance vectors.
//!
//! `_AssignSymmetryClasses` compares those vectors after rendering each entry with
//! "%.4f", so the rounding is part of the definition rather than an artefact. The same
//! equivalence is reproduced here by comparing the exactly-rounded scaled integers,
//! which is what that rendering encodes.
std::vector<int> assign_symmetry_classes(const RDKit::ROMol& mol, unsigned int num_atoms, unsigned int cutoff) {
  const double* distances  = RDKit::MolOps::getDistanceMat(mol,
                                                          /*useBO=*/true,
                                                          /*useAtomWts=*/false,
                                                          /*force=*/true,
                                                          "Balaban");
  const size_t  key_length = std::min<size_t>(cutoff, num_atoms);

  std::map<std::vector<std::int64_t>, int> class_by_key;
  std::vector<int>                         symmetry_classes(num_atoms, 0);
  std::vector<double>                      row(num_atoms);
  std::vector<std::int64_t>                key(key_length);

  for (unsigned int atom_idx = 0; atom_idx < num_atoms; ++atom_idx) {
    std::copy_n(distances + static_cast<size_t>(atom_idx) * num_atoms, num_atoms, row.begin());
    std::sort(row.begin(), row.end());
    for (size_t i = 0; i < key_length; ++i) {
      key[i] = scaled_to_four_decimals(row[i]);
    }
    // Classes are numbered by first appearance, as RDKit's keysSeen list does.
    const auto [entry, unused] = class_by_key.try_emplace(key, static_cast<int>(class_by_key.size()) + 1);
    symmetry_classes[atom_idx] = entry->second;
  }
  return symmetry_classes;
}

//! Asymmetric double sigmoidal function used to map each QED property onto [0, 1].
double asymmetric_double_sigmoidal(double x, const std::array<double, 7>& p) {
  const double a = p[0], b = p[1], c = p[2], d = p[3], e = p[4], f = p[5], dmax = p[6];
  const double exp1 = 1.0 + std::exp(-1.0 * (x - c + d / 2.0) / e);
  const double exp2 = 1.0 + std::exp(-1.0 * (x - c - d / 2.0) / f);
  return (a + b / exp1 * (1.0 - 1.0 / exp2)) / dmax;
}

//! Compiles the QED SMARTS tables once, on first use.
struct QedPatterns {
  std::vector<std::unique_ptr<RDKit::ROMol>> acceptors;
  std::vector<std::unique_ptr<RDKit::ROMol>> alerts;
  std::unique_ptr<RDKit::ROMol>              aliphatic_rings;

  QedPatterns() {
    for (const char* smarts : cuik_molmaker::qed_data::ACCEPTOR_SMARTS) {
      acceptors.emplace_back(RDKit::SmartsToMol(smarts));
    }
    for (const char* smarts : cuik_molmaker::qed_data::STRUCTURAL_ALERT_SMARTS) {
      alerts.emplace_back(RDKit::SmartsToMol(smarts));
    }
    aliphatic_rings.reset(RDKit::SmartsToMol(cuik_molmaker::qed_data::ALIPHATIC_RINGS_SMARTS));
  }
};

const QedPatterns& qed_patterns() {
  static const QedPatterns patterns;
  return patterns;
}

}  // namespace

double bertz_ct(const RDKit::ROMol& mol, unsigned int cutoff) {
  const unsigned int num_atoms = mol.getNumAtoms();
  if (num_atoms < 2) {
    return 0.0;
  }

  // Neighbour lists are deduplicated and sorted, as `_CreateBondDictEtc` does.
  std::vector<std::vector<unsigned int>>                  neighbors(num_atoms);
  std::map<std::pair<unsigned int, unsigned int>, double> bond_orders;
  for (const RDKit::Bond* bond : mol.bonds()) {
    const unsigned int begin                                  = bond->getBeginAtomIdx();
    const unsigned int end                                    = bond->getEndAtomIdx();
    bond_orders[{std::min(begin, end), std::max(begin, end)}] = bertz_bond_order(*bond);
    neighbors[begin].push_back(end);
    neighbors[end].push_back(begin);
  }
  for (auto& list : neighbors) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }

  const std::vector<int> symmetry_classes = assign_symmetry_classes(mol, num_atoms, cutoff);

  // Connection keys are either a bonded pair or a neighbour-hinge-neighbour triple; the
  // leading element records which, so the two kinds never collide.
  std::map<std::array<int, 4>, double> connection_counts;
  std::map<int, double>                atom_type_counts;

  for (unsigned int atom_idx = 0; atom_idx < num_atoms; ++atom_idx) {
    atom_type_counts[mol.getAtomWithIdx(atom_idx)->getAtomicNum()] += 1.0;

    const int    hinge_class    = symmetry_classes[atom_idx];
    const auto&  atom_neighbors = neighbors[atom_idx];
    const size_t num_neighbors  = atom_neighbors.size();

    for (size_t i = 0; i < num_neighbors; ++i) {
      const unsigned int neighbor_i       = atom_neighbors[i];
      const int          neighbor_i_class = symmetry_classes[neighbor_i];
      const double bond_i_order = bond_orders.at({std::min(atom_idx, neighbor_i), std::max(atom_idx, neighbor_i)});

      if (bond_i_order > 1.0 && neighbor_i > atom_idx) {
        const std::array<int, 4> key{2,
                                     std::min(hinge_class, neighbor_i_class),
                                     std::max(hinge_class, neighbor_i_class),
                                     0};
        connection_counts[key] += bond_i_order * (bond_i_order - 1.0) / 2.0;
      }

      for (size_t j = i + 1; j < num_neighbors; ++j) {
        const unsigned int neighbor_j       = atom_neighbors[j];
        const int          neighbor_j_class = symmetry_classes[neighbor_j];
        const double bond_j_order = bond_orders.at({std::min(atom_idx, neighbor_j), std::max(atom_idx, neighbor_j)});

        const std::array<int, 4> key{3,
                                     std::min(neighbor_i_class, neighbor_j_class),
                                     hinge_class,
                                     std::max(neighbor_i_class, neighbor_j_class)};
        connection_counts[key] += bond_i_order * bond_j_order;
      }
    }
  }

  std::vector<double> atom_type_list;
  atom_type_list.reserve(atom_type_counts.size());
  for (const auto& [atomic_num, count] : atom_type_counts) {
    atom_type_list.push_back(count);
  }
  const double atom_type_ie = num_atoms * info_entropy(atom_type_list);

  // An empty connection set contributes nothing: RDKit substitutes a single count of 1,
  // whose entropy and log are both zero.
  double connection_ie = 0.0;
  if (!connection_counts.empty()) {
    std::vector<double> connection_list;
    connection_list.reserve(connection_counts.size());
    double total_connections = 0.0;
    for (const auto& [key, count] : connection_counts) {
      connection_list.push_back(count);
      total_connections += count;
    }
    connection_ie = total_connections * (info_entropy(connection_list) + std::log2(total_connections));
  }

  return atom_type_ie + connection_ie;
}

double qed_weights_mean(const RDKit::ROMol& mol) {
  const QedPatterns& patterns = qed_patterns();

  // QED.properties strips hydrogens first, so every term below sees the same molecule.
  const std::unique_ptr<RDKit::ROMol> stripped(RDKit::MolOps::removeHs(mol));

  double crippen_logp = 0.0;
  double crippen_mr   = 0.0;
  RDKit::Descriptors::calcCrippenDescriptors(*stripped, crippen_logp, crippen_mr);

  // HBA sums every match, so it needs the same uniquify/maxMatches defaults as
  // Mol.GetSubstructMatches; ALERTS only asks whether a pattern matches at all.
  RDKit::SubstructMatchParameters count_params;
  RDKit::SubstructMatchParameters exists_params;
  exists_params.maxMatches = 1;

  double num_acceptor_matches = 0.0;
  for (const auto& acceptor : patterns.acceptors) {
    num_acceptor_matches += static_cast<double>(RDKit::SubstructMatch(*stripped, *acceptor, count_params).size());
  }

  double num_alerts = 0.0;
  for (const auto& alert : patterns.alerts) {
    if (!RDKit::SubstructMatch(*stripped, *alert, exists_params).empty()) {
      num_alerts += 1.0;
    }
  }

  // AROM counts rings that survive deleting every aliphatic ring atom, which is not the
  // same as NumAromaticRings; QED.properties documents the discrepancy.
  const std::unique_ptr<RDKit::ROMol> aromatic_only(RDKit::deleteSubstructs(*stripped, *patterns.aliphatic_rings));
  RDKit::VECT_INT_VECT                rings;
  const double num_aromatic_rings = static_cast<double>(RDKit::MolOps::findSSSR(*aromatic_only, rings));

  const std::array<double, 8> properties{
    RDKit::Descriptors::calcAMW(*stripped),
    crippen_logp,
    num_acceptor_matches,
    static_cast<double>(RDKit::Descriptors::calcNumHBD(*stripped)),
    RDKit::Descriptors::calcTPSA(*stripped),
    static_cast<double>(RDKit::Descriptors::calcNumRotatableBonds(*stripped, RDKit::Descriptors::Strict)),
    num_aromatic_rings,
    num_alerts,
  };

  double weighted_log_sum = 0.0;
  double weight_total     = 0.0;
  for (size_t i = 0; i < properties.size(); ++i) {
    const double desirability = asymmetric_double_sigmoidal(properties[i], cuik_molmaker::qed_data::ADS_PARAMETERS[i]);
    const double weight       = cuik_molmaker::qed_data::WEIGHT_MEAN[i];
    weighted_log_sum += weight * std::log(desirability);
    weight_total += weight;
  }
  return std::exp(weighted_log_sum / weight_total);
}

double balaban_j(const RDKit::ROMol& mol) {
  const unsigned int num_atoms = mol.getNumAtoms();
  const unsigned int num_bonds = mol.getNumBonds();
  if (num_atoms == 0) {
    return 0.0;
  }

  // Row sums of the bond-order distance matrix are Balaban's vertex distance degrees.
  const double* distances =
    RDKit::MolOps::getDistanceMat(mol, /*useBO=*/true, /*useAtomWts=*/false, /*force=*/true, "Balaban");
  std::vector<double> vertex_degrees(num_atoms, 0.0);
  for (unsigned int i = 0; i < num_atoms; ++i) {
    double row_sum = 0.0;
    for (unsigned int j = 0; j < num_atoms; ++j) {
      row_sum += distances[static_cast<size_t>(i) * num_atoms + j];
    }
    vertex_degrees[i] = row_sum;
  }

  // The adjacency test in RDKit's loop admits exactly the bonded pairs, once each.
  double sum = 0.0;
  for (const RDKit::Bond* bond : mol.bonds()) {
    sum += 1.0 / std::sqrt(vertex_degrees[bond->getBeginAtomIdx()] * vertex_degrees[bond->getEndAtomIdx()]);
  }

  const int cyclomatic_number = static_cast<int>(num_bonds) - static_cast<int>(num_atoms) + 1;
  if (cyclomatic_number + 1 == 0) {
    return 0.0;
  }
  return static_cast<double>(num_bonds) / (cyclomatic_number + 1) * sum;
}

namespace {

//! Fragment scores keyed by Morgan bit id, loaded once from the generated table.
struct SaFragmentTable {
  std::vector<std::uint32_t> keys;         //!< sorted, unique
  std::vector<std::uint16_t> score_index;  //!< parallel to `keys`
  std::vector<double>        scores;       //!< distinct score values

  //! Score for one Morgan bit, or the -4 penalty RDKit uses for unseen fragments.
  double lookup(std::uint32_t key) const {
    const auto it = std::lower_bound(keys.begin(), keys.end(), key);
    if (it == keys.end() || *it != key) {
      return -4.0;
    }
    return scores[score_index[static_cast<size_t>(it - keys.begin())]];
  }
};

std::string& sa_fragment_path() {
  static std::string path;
  return path;
}

//! Reads a little-endian array of `count` elements into a vector.
template <typename T> void read_array(std::ifstream& stream, std::vector<T>& out, size_t count, const char* what) {
  out.resize(count);
  stream.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!stream) {
    throw std::runtime_error(std::string("SA score fragment table truncated while reading ") + what);
  }
}

const SaFragmentTable& sa_fragment_table() {
  static const SaFragmentTable table = [] {
    const std::string& path = sa_fragment_path();
    if (path.empty()) {
      throw std::runtime_error("SA score fragment table path is unset; the package sets it on import");
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
      throw std::runtime_error("Could not open SA score fragment table: " + path);
    }

    char magic[8] = {};
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, "CUIKSAS1", sizeof(magic)) != 0) {
      throw std::runtime_error("Not an SA score fragment table: " + path);
    }

    std::uint64_t header[2] = {0, 0};
    stream.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!stream) {
      throw std::runtime_error("SA score fragment table header truncated: " + path);
    }

    // Check the counts against the file before allocating: they come from the file, so an
    // absurd count would otherwise be a multi-exabyte resize.
    const std::streamoff header_bytes = sizeof(magic) + sizeof(header);
    stream.seekg(0, std::ios::end);
    const std::streamoff file_size = stream.tellg();
    stream.seekg(header_bytes, std::ios::beg);
    if (!stream || file_size < header_bytes) {
      throw std::runtime_error("SA score fragment table is too short: " + path);
    }

    const std::uint64_t entry_bytes = sizeof(std::uint32_t) + sizeof(std::uint16_t);
    const std::uint64_t payload     = static_cast<std::uint64_t>(file_size - header_bytes);
    const std::uint64_t num_entries = header[0];
    const std::uint64_t num_scores  = header[1];
    if (num_entries > payload / entry_bytes || num_scores > payload / sizeof(double) ||
        num_entries * entry_bytes + num_scores * sizeof(double) != payload) {
      throw std::runtime_error("SA score fragment table has inconsistent counts: " + path);
    }

    SaFragmentTable loaded;
    read_array(stream, loaded.keys, num_entries, "keys");
    read_array(stream, loaded.score_index, num_entries, "score indices");
    read_array(stream, loaded.scores, num_scores, "score table");

    // lookup() binary-searches the keys and indexes the score table, so an unsorted key
    // array or an out-of-range index would return a wrong score rather than fail.
    if (!std::is_sorted(loaded.keys.begin(), loaded.keys.end())) {
      throw std::runtime_error("SA score fragment table keys are not sorted: " + path);
    }
    const std::size_t score_count = loaded.scores.size();
    if (std::any_of(loaded.score_index.begin(), loaded.score_index.end(), [score_count](std::uint16_t index) {
          return index >= score_count;
        })) {
      throw std::runtime_error("SA score fragment table has an out-of-range score index: " + path);
    }
    return loaded;
  }();
  return table;
}

//! Number of chiral centres RDKit's legacy perception reports, assigned and unassigned.
unsigned int count_chiral_centers(const RDKit::ROMol& mol) {
  // assignStereochemistry rewrites atom properties, so it runs on a private copy.
  RDKit::RWMol working(mol);
  for (RDKit::Atom* atom : working.atoms()) {
    atom->clearProp(RDKit::common_properties::_ChiralityPossible);
  }
  RDKit::MolOps::assignStereochemistry(working,
                                       /*cleanIt=*/true,
                                       /*force=*/true,
                                       /*flagPossibleStereoCenters=*/true);

  unsigned int centers = 0;
  for (const RDKit::Atom* atom : working.atoms()) {
    if (atom->hasProp(RDKit::common_properties::_CIPCode) ||
        atom->hasProp(RDKit::common_properties::_ChiralityPossible)) {
      ++centers;
    }
  }
  return centers;
}

}  // namespace

void set_sa_score_fragment_path(const std::string& path) {
  sa_fragment_path() = path;
}

double sa_score(const RDKit::ROMol& mol) {
  const unsigned int num_atoms = mol.getNumAtoms();
  if (num_atoms == 0) {
    throw std::invalid_argument("SA score is not defined for a molecule with no atoms");
  }
  const SaFragmentTable& table = sa_fragment_table();

  const std::unique_ptr<RDKit::SparseIntVect<std::uint32_t>> fingerprint(
    RDKit::MorganFingerprints::getFingerprint(mol, /*radius=*/2));
  const auto& nonzero = fingerprint->getNonzeroElements();

  double fragment_total = 0.0;
  double num_fragments  = 0.0;
  for (const auto& [bit, count] : nonzero) {
    num_fragments += count;
    fragment_total += table.lookup(bit) * count;
  }
  const double fragment_score = fragment_total / num_fragments;

  const RDKit::RingInfo* ring_info      = mol.getRingInfo();
  bool                   has_macrocycle = false;
  for (const auto& ring : ring_info->atomRings()) {
    if (ring.size() > 8) {
      has_macrocycle = true;
      break;
    }
  }

  const double feature_score =
    -((std::pow(static_cast<double>(num_atoms), 1.005) - num_atoms) + std::log10(count_chiral_centers(mol) + 1.0) +
      std::log10(RDKit::Descriptors::calcNumSpiroAtoms(mol) + 1.0) +
      std::log10(RDKit::Descriptors::calcNumBridgeheadAtoms(mol) + 1.0) + (has_macrocycle ? std::log10(2.0) : 0.0));

  // Highly symmetrical molecules repeat few distinct fragments and are easier to make.
  const size_t num_distinct_bits = nonzero.size();
  const double symmetry_score =
    num_atoms > num_distinct_bits ?
      std::log(static_cast<double>(num_atoms) / static_cast<double>(num_distinct_bits)) * 0.5 :
      0.0;

  constexpr double raw_min = -4.0;
  constexpr double raw_max = 2.5;
  const double     raw     = fragment_score + feature_score + symmetry_score;
  double           sascore = 11.0 - (raw - raw_min + 1.0) / (raw_max - raw_min) * 9.0;

  // sascorer.py subtracts 9 here, which makes log() diverge at the branch boundary and
  // slams scores just above 8 down to 1; see RDKit github #8251 and PR #9501.
  if (sascore > 8.0) {
    sascore = 8.0 + std::log(sascore - 7.0);
  }
  return std::clamp(sascore, 1.0, 10.0);
}

namespace {

//! One column of the output table: how to compute it from a parsed molecule.
using DescriptorFn = double (*)(const RDKit::ROMol&);

// Descriptors whose own signature carries defaulted parameters, or returns its result
// through an out-parameter, need a thin adapter to match DescriptorFn.

double descriptor_mol_wt(const RDKit::ROMol& mol) {
  return RDKit::Descriptors::calcAMW(mol);
}
double descriptor_tpsa(const RDKit::ROMol& mol) {
  return RDKit::Descriptors::calcTPSA(mol);
}
double descriptor_hall_kier_alpha(const RDKit::ROMol& mol) {
  return RDKit::Descriptors::calcHallKierAlpha(mol);
}
double descriptor_mol_log_p(const RDKit::ROMol& mol) {
  double logp = 0.0, mr = 0.0;
  RDKit::Descriptors::calcCrippenDescriptors(mol, logp, mr);
  return logp;
}
double descriptor_mol_mr(const RDKit::ROMol& mol) {
  double logp = 0.0, mr = 0.0;
  RDKit::Descriptors::calcCrippenDescriptors(mol, logp, mr);
  return mr;
}
double descriptor_bertz_ct(const RDKit::ROMol& mol) {
  return bertz_ct(mol);
}

const std::vector<std::pair<std::string, DescriptorFn>>& descriptor_table() {
  static const std::vector<std::pair<std::string, DescriptorFn>> table{
    {          "qed",           qed_weights_mean},
    {        "MolWt",          descriptor_mol_wt},
    {     "BalabanJ",                  balaban_j},
    {      "BertzCT",        descriptor_bertz_ct},
    {"HallKierAlpha", descriptor_hall_kier_alpha},
    {         "TPSA",            descriptor_tpsa},
    {      "MolLogP",       descriptor_mol_log_p},
    {        "MolMR",          descriptor_mol_mr},
    {      "SAScore",                   sa_score},
  };
  return table;
}

}  // namespace

std::vector<std::string> list_all_molecular_descriptors() {
  std::vector<std::string> names;
  names.reserve(descriptor_table().size());
  for (const auto& [name, fn] : descriptor_table()) {
    names.push_back(name);
  }
  return names;
}

namespace {

//! Turns one serialized molecule into an ROMol, or nullptr if it cannot be read.
using MolParser = std::unique_ptr<RDKit::ROMol> (*)(const std::string&);

std::unique_ptr<RDKit::ROMol> mol_from_smiles(const std::string& smiles) {
  try {
    return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  } catch (const std::exception&) {
    return nullptr;
  }
}

std::unique_ptr<RDKit::ROMol> mol_from_binary(const std::string& pickle) {
  // MolPickler trusts the stream and reads lengths straight out of it, so random bytes
  // can fault inside RDKit rather than raising. The stream opens with the endian marker.
  std::int32_t endian_marker = 0;
  if (pickle.size() < sizeof(endian_marker)) {
    return nullptr;
  }
  std::memcpy(&endian_marker, pickle.data(), sizeof(endian_marker));
  if (endian_marker != RDKit::MolPickler::endianId) {
    return nullptr;
  }

  auto mol = std::make_unique<RDKit::RWMol>();
  try {
    RDKit::MolPickler::molFromPickle(pickle, *mol);
  } catch (...) {
    return nullptr;
  }
  // Several descriptors read RingInfo directly, which older pickles may omit.
  if (!mol->getRingInfo()->isInitialized()) {
    RDKit::MolOps::findSSSR(*mol);
  }
  return mol;
}

//! Resolves descriptor names to their implementations, preserving the requested order.
std::vector<DescriptorFn> resolve_descriptors(const std::vector<std::string>& descriptor_names) {
  std::vector<DescriptorFn> requested;
  requested.reserve(descriptor_names.size());
  for (const std::string& name : descriptor_names) {
    const auto& table = descriptor_table();
    const auto  it =
      std::find_if(table.begin(), table.end(), [&name](const auto& entry) { return entry.first == name; });
    if (it == table.end()) {
      throw std::invalid_argument("Unknown molecular descriptor: " + name);
    }
    requested.push_back(it->second);
  }
  return requested;
}

//! Shared body of both batch entry points; `inputs` are already plain strings so the
//! workers never touch Python objects.
py::array_t<double> compute_descriptor_table(const std::vector<std::string>& inputs,
                                             const std::vector<std::string>& descriptor_names,
                                             int                             num_threads,
                                             MolParser                       parse_mol) {
  const std::vector<DescriptorFn> requested = resolve_descriptors(descriptor_names);

  const size_t num_mols = inputs.size();
  const size_t num_desc = requested.size();

  std::unique_ptr<double[]> data(new double[num_mols * num_desc]);
  double* const             raw = data.get();

  // Warm the shared tables before releasing the GIL so the workers never race to
  // initialise them, and so a missing SA table raises in Python rather than in a thread.
  if (!requested.empty()) {
    qed_patterns();
  }
  if (std::find(descriptor_names.begin(), descriptor_names.end(), "SAScore") != descriptor_names.end()) {
    sa_fragment_table();
  }

  {
    py::gil_scoped_release release;

    // More workers than cores buys nothing for this work, and an unbounded request
    // would try to create that many OS threads.
    const unsigned int available = std::max(1u, std::thread::hardware_concurrency());
    unsigned int workers = num_threads > 0 ? std::min(static_cast<unsigned int>(num_threads), available) : available;
    workers              = std::max(1u, std::min<unsigned int>(workers, static_cast<unsigned int>(num_mols)));

    const auto compute_range = [&](size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
        double* const                       out = raw + row * num_desc;
        const std::unique_ptr<RDKit::ROMol> mol = parse_mol(inputs[row]);
        if (!mol) {
          std::fill_n(out, num_desc, std::numeric_limits<double>::quiet_NaN());
          continue;
        }
        for (size_t col = 0; col < num_desc; ++col) {
          try {
            out[col] = requested[col](*mol);
          } catch (const std::exception&) {
            out[col] = std::numeric_limits<double>::quiet_NaN();
          }
        }
      }
    };

    if (workers <= 1 || num_mols == 0) {
      compute_range(0, num_mols);
    } else {
      // jthread joins on destruction, so failing to create the Nth worker still joins the
      // first N-1 while unwinding instead of terminating on a joinable thread.
      std::vector<std::jthread> pool;
      pool.reserve(workers);
      const size_t chunk = (num_mols + workers - 1) / workers;
      for (unsigned int w = 0; w < workers; ++w) {
        const size_t begin = std::min(num_mols, static_cast<size_t>(w) * chunk);
        const size_t end   = std::min(num_mols, begin + chunk);
        if (begin < end) {
          pool.emplace_back(compute_range, begin, end);
        }
      }
    }
  }

  const int64_t dims[2] = {static_cast<int64_t>(num_mols), static_cast<int64_t>(num_desc)};
  return py_array_from_array(std::move(data), dims, 2);
}

}  // namespace

py::array_t<double> batch_molecular_descriptors(const std::vector<std::string>& smiles_list,
                                                const std::vector<std::string>& descriptor_names,
                                                int                             num_threads) {
  return compute_descriptor_table(smiles_list, descriptor_names, num_threads, mol_from_smiles);
}

py::array_t<double> batch_molecular_descriptors_from_binary(const std::vector<std::string>& mol_binaries,
                                                            const std::vector<std::string>& descriptor_names,
                                                            int                             num_threads) {
  return compute_descriptor_table(mol_binaries, descriptor_names, num_threads, mol_from_binary);
}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "descriptor_context.h"

// clang-format off
// BCUT.h declares BCUT2D only when RDK_HAS_EIGEN3 is defined, which RDConfig.h does, so
// include sorting must not reorder these two.
#include <RDGeneral/RDConfig.h>
#include <GraphMol/Descriptors/BCUT.h>
// clang-format on

#include <DataStructs/SparseIntVect.h>
#include <GraphMol/Descriptors/Crippen.h>
#include <GraphMol/Descriptors/MolSurf.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PartialCharges/GasteigerCharges.h>
#include <GraphMol/PeriodicTable.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace cuik_molmaker {

namespace {

//! Mirrors `EState.GetPrincipleQuantumNumber`.
int principal_quantum_number(unsigned int atomic_num) {
  if (atomic_num <= 2)
    return 1;
  if (atomic_num <= 10)
    return 2;
  if (atomic_num <= 18)
    return 3;
  if (atomic_num <= 36)
    return 4;
  if (atomic_num <= 54)
    return 5;
  if (atomic_num <= 86)
    return 6;
  return 7;
}

// Bin edges of rdkit.Chem.EState.EState_VSA; a value v falls in bin bisect_right(edges, v).
constexpr std::array<double, 10> ESTATE_BIN_EDGES{-0.39, 0.29, 0.717, 1.165, 1.54, 1.807, 2.05, 4.69, 9.17, 15.0};
constexpr std::array<double, 9>  VSA_BIN_EDGES{4.78, 5.0, 5.41, 5.74, 6.0, 6.07, 6.45, 7.0, 11.0};

template <size_t N> size_t bisect_right(const std::array<double, N>& edges, double value) {
  return static_cast<size_t>(std::upper_bound(edges.begin(), edges.end(), value) - edges.begin());
}

//! The Morgan generator `rdFingerprintGenerator.GetMorganGenerator(radius)` builds.
//!
//! One per thread and radius: a generator carries its invariant generators, and sharing
//! one across threads is not documented as safe.
const RDKit::FingerprintGenerator<std::uint32_t>& morgan_generator(unsigned int radius) {
  thread_local std::array<std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>, 4> generators;
  auto& generator = generators.at(radius);
  if (!generator) {
    generator.reset(
      RDKit::MorganFingerprint::getMorganGenerator<std::uint32_t>(radius,
                                                                  /*countSimulation=*/false,
                                                                  /*includeChirality=*/false,
                                                                  /*useBondTypes=*/true,
                                                                  /*onlyNonzeroInvariants=*/false,
                                                                  /*includeRedundantEnvironments=*/false));
  }
  return *generator;
}

}  // namespace

const double* bond_order_distance_matrix(const RDKit::ROMol& mol) {
  return RDKit::MolOps::getDistanceMat(mol, /*useBO=*/true, /*useAtomWts=*/false, /*force=*/true, "Balaban");
}

const double* DescriptorContext::bond_order_distances() {
  if (bond_order_distances_ == nullptr) {
    bond_order_distances_ = bond_order_distance_matrix(mol_);
  }
  return bond_order_distances_;
}

const double* DescriptorContext::topological_distances() {
  if (topological_distances_ == nullptr) {
    topological_distances_ = RDKit::MolOps::getDistanceMat(mol_, /*useBO=*/false, /*useAtomWts=*/false);
  }
  return topological_distances_;
}

const std::pair<double, double>& DescriptorContext::crippen() {
  if (!crippen_) {
    double logp = 0.0, mr = 0.0;
    RDKit::Descriptors::calcCrippenDescriptors(mol_, logp, mr);
    crippen_.emplace(logp, mr);
  }
  return *crippen_;
}

const std::vector<double>& DescriptorContext::estate_indices() {
  if (!estate_indices_) {
    // Every expression keeps the operand order of EState.EStateIndices, so the result is
    // bit-for-bit what the Python loop produces.
    const unsigned int          num_atoms = mol_.getNumAtoms();
    const RDKit::PeriodicTable* table     = RDKit::PeriodicTable::getTable();

    std::vector<double> intrinsic(num_atoms, 0.0);
    for (const RDKit::Atom* atom : mol_.atoms()) {
      const unsigned int degree = atom->getDegree();
      if (degree == 0) {
        continue;
      }
      const unsigned int atomic_num = atom->getAtomicNum();
      const int valence_electrons   = table->getNouterElecs(atomic_num) - static_cast<int>(atom->getTotalNumHs());
      const int principal           = principal_quantum_number(atomic_num);
      intrinsic[atom->getIdx()]     = (4.0 / (principal * principal) * valence_electrons + 1.0) / degree;
    }

    const double*       distances = topological_distances();
    std::vector<double> accumulated(num_atoms, 0.0);
    for (unsigned int i = 0; i < num_atoms; ++i) {
      for (unsigned int j = i + 1; j < num_atoms; ++j) {
        const double path = distances[static_cast<size_t>(i) * num_atoms + j] + 1.0;
        if (path < 1e6) {
          const double perturbation = (intrinsic[i] - intrinsic[j]) / (path * path);
          accumulated[i] += perturbation;
          accumulated[j] -= perturbation;
        }
      }
    }
    for (unsigned int i = 0; i < num_atoms; ++i) {
      accumulated[i] = accumulated[i] + intrinsic[i];
    }
    estate_indices_ = std::move(accumulated);
  }
  return *estate_indices_;
}

const std::vector<double>& DescriptorContext::labute_atom_contribs() {
  if (!labute_atom_contribs_) {
    std::vector<double> contribs(mol_.getNumAtoms(), 0.0);
    double              hydrogen_contrib = 0.0;
    RDKit::Descriptors::getLabuteAtomContribs(mol_, contribs, hydrogen_contrib, /*includeHs=*/true);
    labute_atom_contribs_ = std::move(contribs);
  }
  return *labute_atom_contribs_;
}

const std::array<double, 11>& DescriptorContext::estate_vsa() {
  if (!estate_vsa_) {
    const std::vector<double>& estate = estate_indices();
    const std::vector<double>& volume = labute_atom_contribs();
    std::array<double, 11>     bins{};
    for (size_t i = 0; i < estate.size(); ++i) {
      bins[bisect_right(ESTATE_BIN_EDGES, estate[i])] += volume[i];
    }
    estate_vsa_ = bins;
  }
  return *estate_vsa_;
}

const std::array<double, 10>& DescriptorContext::vsa_estate() {
  if (!vsa_estate_) {
    const std::vector<double>& estate = estate_indices();
    const std::vector<double>& volume = labute_atom_contribs();
    std::array<double, 10>     bins{};
    for (size_t i = 0; i < estate.size(); ++i) {
      bins[bisect_right(VSA_BIN_EDGES, volume[i])] += estate[i];
    }
    vsa_estate_ = bins;
  }
  return *vsa_estate_;
}

const std::pair<double, double>& DescriptorContext::gasteiger_extrema() {
  if (!gasteiger_extrema_) {
    RDKit::computeGasteigerCharges(mol_, /*nIter=*/12, /*throwOnParamFailure=*/false);
    // _ChargeDescriptors folds with Python's min(charge, lowest) and max(charge, highest),
    // which keep their first argument unless the second compares strictly better. The
    // ternaries below are that rule, so a NaN charge propagates or drops out the same way.
    double lowest = 500.0, highest = -500.0;
    for (const RDKit::Atom* atom : mol_.atoms()) {
      const double charge = atom->getProp<double>(RDKit::common_properties::_GasteigerCharge);
      lowest              = lowest < charge ? lowest : charge;
      highest             = highest > charge ? highest : charge;
    }
    gasteiger_extrema_.emplace(lowest, highest);
  }
  return *gasteiger_extrema_;
}

size_t DescriptorContext::morgan_environment_count(unsigned int radius) {
  if (radius < 1 || radius > 3) {
    throw std::invalid_argument("Morgan environment counts are kept for radii 1 to 3");
  }
  auto& count = morgan_environment_counts_[radius];
  if (!count) {
    const std::unique_ptr<RDKit::SparseIntVect<std::uint32_t>> fingerprint(
      morgan_generator(radius).getSparseCountFingerprint(mol_));
    count = fingerprint->getNonzeroElements().size();
  }
  return *count;
}

const std::vector<double>& DescriptorContext::bcut2d() {
  if (!bcut2d_) {
    bcut2d_ = RDKit::Descriptors::BCUT2D(mol_);
  }
  return *bcut2d_;
}

}  // namespace cuik_molmaker

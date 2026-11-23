// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file Per-molecule cache of the intermediates that several whole-molecule descriptors
//!       share. Internal to the descriptor table in molecular_descriptors.cpp.

#pragma once

#include <GraphMol/ROMol.h>

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace cuik_molmaker {

//! Intermediates that several descriptors of one molecule share, each computed on first use.
//!
//! A table row asks for many descriptors of the same molecule, and several start from the
//! same expensive quantity. One context serves one molecule on one thread, so nothing here
//! is synchronised. The molecule must outlive the context.
class DescriptorContext {
 public:
  explicit DescriptorContext(const RDKit::ROMol& mol) : mol_(mol) {}

  DescriptorContext(const DescriptorContext&)            = delete;
  DescriptorContext& operator=(const DescriptorContext&) = delete;

  const RDKit::ROMol& mol() const { return mol_; }

  //! Bond-order-weighted topological distances, as BalabanJ and BertzCT read them;
  //! row-major, `getNumAtoms()` squared. Owned by the molecule's computed properties.
  const double* bond_order_distances();

  //! Unweighted topological distances, as `Chem.GetDistanceMatrix(mol)`; unreachable
  //! pairs hold RDKit's 1e8 sentinel. Same layout and ownership as above.
  const double* topological_distances();

  //! Wildman-Crippen logP and molar refractivity, which RDKit computes in one pass.
  const std::pair<double, double>& crippen();

  //! E-state index of every atom, in atom order, as `EState.EStateIndices`.
  const std::vector<double>& estate_indices();

  //! Labute ASA contribution of every atom, hydrogens included, in atom order.
  const std::vector<double>& labute_atom_contribs();

  //! Labute ASA summed within each E-state bin: `EState_VSA1` .. `EState_VSA11`.
  const std::array<double, 11>& estate_vsa();

  //! E-state summed within each Labute ASA bin: `VSA_EState1` .. `VSA_EState10`.
  const std::array<double, 10>& vsa_estate();

  //! Lowest and highest Gasteiger charge, reduced exactly as `Descriptors._ChargeDescriptors`
  //! does, including its starting values and its handling of NaN charges.
  const std::pair<double, double>& gasteiger_extrema();

  //! Number of distinct Morgan environments of the given radius (1 to 3), as counted by
  //! `Descriptors._FingerprintDensity` from a sparse count fingerprint.
  size_t morgan_environment_count(unsigned int radius);

  //! The eight BCUT2D eigenvalues, in RDKit's `BCUT2D_*` order.
  const std::vector<double>& bcut2d();

 private:
  const RDKit::ROMol&                      mol_;
  const double*                            bond_order_distances_  = nullptr;
  const double*                            topological_distances_ = nullptr;
  std::optional<std::pair<double, double>> crippen_;
  std::optional<std::vector<double>>       estate_indices_;
  std::optional<std::vector<double>>       labute_atom_contribs_;
  std::optional<std::array<double, 11>>    estate_vsa_;
  std::optional<std::array<double, 10>>    vsa_estate_;
  std::optional<std::pair<double, double>> gasteiger_extrema_;
  std::array<std::optional<size_t>, 4>     morgan_environment_counts_;
  std::optional<std::vector<double>>       bcut2d_;
};

//! The bond-order distance matrix, recomputed, for callers without a context.
const double* bond_order_distance_matrix(const RDKit::ROMol& mol);

}  // namespace cuik_molmaker

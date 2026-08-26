// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This header declares whole-molecule descriptors that RDKit implements only in
//!       Python, plus the batch entry point that computes a descriptor table across
//!       many SMILES at once. Implementations are in molecular_descriptors.cpp.
//!
//!       RDKit's C++ Descriptors library covers most of `Descriptors._descList`, but
//!       `BertzCT` and `qed` exist only as Python. They dominate the runtime of a
//!       selected-descriptor workload, so they are ported here.

#pragma once

#include <GraphMol/ROMol.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "export.h"

namespace py = pybind11;

//! Bertz CT topological complexity index.
//!
//! Port of `rdkit.Chem.GraphDescriptors.BertzCT` (version 2.0.0), which uses consistent
//! aromatic bond orders rather than a Kekule structure.
//!
//! @param mol Molecule to score; hydrogens are treated as RDKit presents them
//! @param cutoff Number of nearest-neighbour distances used to decide whether two atoms
//!               belong to the same symmetry class. Matches RDKit's default of 100.
//! @return The complexity index, or 0 for molecules with fewer than two atoms
CUIK_EXPORT double bertz_ct(const RDKit::ROMol& mol, unsigned int cutoff = 100);

//! Quantitative Estimate of Drug-likeness, using the mean property weights.
//!
//! Port of `rdkit.Chem.QED.qed` with `w=WEIGHT_MEAN`. Hydrogens are removed first, as
//! `QED.properties` does, so the result does not depend on how they were specified.
//!
//! @param mol Molecule to score
//! @return QED in [0, 1]
CUIK_EXPORT double qed_weights_mean(const RDKit::ROMol& mol);

//! Balaban's J connectivity index.
//!
//! Port of `rdkit.Chem.GraphDescriptors.BalabanJ`, following Balaban,
//! Chem. Phys. Lett. vol 89, 399-404 (1982).
//!
//! @param mol Molecule to score
//! @return The J value, or 0 when the cyclomatic number degenerates
CUIK_EXPORT double balaban_j(const RDKit::ROMol& mol);

//! Points the SA score at its fragment table; must be called before `sa_score`.
//!
//! The table is read on first use rather than here, so start-up cost is only paid by
//! callers that actually ask for the descriptor.
//!
//! @param path Binary table produced by tools/generate_sa_fragment_table.py
CUIK_EXPORT void set_sa_score_fragment_path(const std::string& path);

//! Synthetic accessibility score, in [1, 10]; higher means harder to make.
//!
//! Port of RDKit's Contrib SA_Score with the normalization fix from RDKit PR #9501:
//! the original smoothing of the hard-to-make end used `log(scaled - 8)`, which
//! diverges at the boundary it is applied from, so a molecule scoring just above 8
//! came out as the easiest possible. This uses `log(scaled - 7)`, continuous at 8.
//!
//! @param mol Molecule to score
//! @return Score clamped to [1, 10]
//! @throws std::invalid_argument if the molecule has no atoms, for which the score is
//!         undefined; the batch entry points report that as NaN for the cell
//! @throws std::runtime_error if the fragment table is missing, unreadable or malformed
CUIK_EXPORT double sa_score(const RDKit::ROMol& mol);

//! Names accepted by `batch_molecular_descriptors`, in a stable order.
CUIK_EXPORT std::vector<std::string> list_all_molecular_descriptors();

//! Computes a descriptor table for a batch of SMILES, one molecule per thread slot.
//!
//! Molecules are independent, so the batch is split across threads with the GIL released.
//! A SMILES that does not parse yields a row of NaN rather than raising, matching the
//! Python `MoleculeFeaturizer`; a descriptor that rejects a molecule yields NaN for that
//! cell alone.
//!
//! @param smiles_list SMILES to featurize, one per output row
//! @param descriptor_names Descriptors to compute, one per output column
//! @param num_threads Worker threads, capped at the hardware concurrency because the work
//!                    is CPU-bound; 0 selects it directly
//! @return Array of shape `(smiles_list.size(), descriptor_names.size())`, dtype float64
//! @throws std::invalid_argument if any descriptor name is unknown
CUIK_EXPORT py::array_t<double> batch_molecular_descriptors(const std::vector<std::string>& smiles_list,
                                                            const std::vector<std::string>& descriptor_names,
                                                            int                             num_threads);

//! Computes a descriptor table for molecules supplied as RDKit binary pickles.
//!
//! A SMILES round-trip is lossy for this purpose: `MolFromSmiles` sanitizes, which removes
//! explicit hydrogens, so descriptors that read the hydrogen-bearing graph -- `BalabanJ`,
//! `BertzCT` and `SAScore` -- disagree with the caller's molecule. Callers holding an
//! `RWMol` should serialize it with `Mol.ToBinary()` and use this entry point.
//!
//! @param mol_binaries Molecules serialised with RDKit's `Mol.ToBinary()`
//! @param descriptor_names Descriptors to compute, one per output column
//! @param num_threads Worker threads, capped at the hardware concurrency because the work
//!                    is CPU-bound; 0 selects it directly
//! @return Array of shape `(mol_binaries.size(), descriptor_names.size())`, dtype float64
//! @throws std::invalid_argument if any descriptor name is unknown
CUIK_EXPORT py::array_t<double> batch_molecular_descriptors_from_binary(
  const std::vector<std::string>& mol_binaries,
  const std::vector<std::string>& descriptor_names,
  int                             num_threads);

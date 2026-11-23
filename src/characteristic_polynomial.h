// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file Characteristic polynomial of a molecular graph, behind the Ipc and AvgIpc descriptors.

#pragma once

#include <GraphMol/ROMol.h>

#include <vector>

namespace cuik_molmaker {

//! |c_0| .. |c_n| of det(xI - A) for the molecule's adjacency matrix A, as doubles.
//!
//! The coefficients are integers and are computed exactly, then each rounded to double
//! once. det(xI - A) is evaluated modulo 62-bit primes at n + 1 points by an elimination
//! ordered along the molecular graph, interpolated, and the residues combined by the Chinese
//! remainder theorem; a bound on the coefficients fixes how many primes that takes.
//!
//! RDKit 2026.03 and earlier compute these with the Le Verrier-Faddeev recursion in
//! floating point: identical for small molecules, wrong above about 65 atoms. RDKit 2026.09.2
//! and later expand from eigenvalues, within about 1e-13.
//!
//! @param mol Molecule whose atoms and bonds define the graph, hydrogens included as present
//! @return n + 1 magnitudes, leading coefficient first; empty for a molecule with no atoms
//! @throws std::range_error past about 3,000 atoms of a typical molecular graph, where the
//!         coefficients outgrow the prime table and Ipc is far beyond the double range anyway
std::vector<double> characteristic_polynomial_magnitudes(const RDKit::ROMol& mol);

}  // namespace cuik_molmaker

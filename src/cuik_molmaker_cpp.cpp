// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This file specifies which functions are exported to Python,
//!       as well as defining `parse_mol` and `get_canonical_atom_order`,
//!       declared in features.h and called from features.cpp and labels.cpp

#include "features.h"
#include "molecular_descriptors.h"

// C++ standard library headers
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// RDKit headers
#include <GraphMol/Atom.h>
#include <GraphMol/Canon.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/new_canon.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <RDGeneral/types.h>

// PyBind headers for use by library to be imported by Python
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// This is necessary to export Python functions in a Python module named cuik_molmaker.
PYBIND11_MODULE(cuik_molmaker_cpp, m) {
  m.doc() = "Cuik MolMaker C++ plugin";  // Python module docstring

  // Functions in features.cpp
  m.def("atom_onehot_feature_names_to_array",
        &atom_onehot_feature_names_to_array,
        "Accepts feature names and returns a NumPy array representing them as integers");
  m.def("atom_float_feature_names_to_array",
        &atom_float_feature_names_to_array,
        "Accepts feature names and returns a NumPy array representing them as integers");
  m.def("bond_feature_names_to_array",
        &bond_feature_names_to_array,
        "Accepts feature names and returns a NumPy array representing them as integers");
  m.def(
    "mol_featurizer",
    &mol_featurizer,
    "Accepts a SMILES string and returns a list of NumPy arrays representing atom and bond features of the molecule.");
  m.def(
    "batch_mol_featurizer",
    &batch_mol_featurizer,
    "Accepts a list of SMILES strings and returns a list of NumPy arrays representing atom and bond features of the molecules.");
  m.def("batch_mol_featurizer_from_binary",
        &batch_mol_featurizer_from_binary,
        "Accepts a list of RDKit binary mol pickles (Mol.ToBinary()) and returns the same arrays as "
        "batch_mol_featurizer. Unlike the SMILES entry point this preserves the caller's exact atom "
        "and bond ordering, and therefore the neighbour-relative chiral tags.");

  m.def("list_all_atom_onehot_features",
        &list_all_atom_onehot_features,
        "Returns a list of all atom one-hot features.");

  m.def("list_all_atom_float_features", &list_all_atom_float_features, "Returns a list of all atom float features.");

  m.def("list_all_bond_features", &list_all_bond_features, "Returns a list of all bond features.");

  // Whole-molecule descriptors (molecular_descriptors.cpp)
  m.def(
    "bertz_ct",
    [](const std::string& smiles, unsigned int cutoff) {
      const std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
      if (!mol) {
        throw std::invalid_argument("Could not parse SMILES: " + smiles);
      }
      return bertz_ct(*mol, cutoff);
    },
    py::arg("smiles"),
    py::arg("cutoff") = 100,
    "Bertz CT topological complexity index for one SMILES string. RDKit implements this "
    "only in Python; this is a port of GraphDescriptors.BertzCT version 2.0.0.");
  m.def(
    "qed_weights_mean",
    [](const std::string& smiles) {
      const std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
      if (!mol) {
        throw std::invalid_argument("Could not parse SMILES: " + smiles);
      }
      return qed_weights_mean(*mol);
    },
    py::arg("smiles"),
    "Quantitative Estimate of Drug-likeness with mean weights, for one SMILES string. "
    "Port of rdkit.Chem.QED.qed; hydrogens are removed first, as QED.properties does.");
  m.def(
    "balaban_j",
    [](const std::string& smiles) {
      const std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
      if (!mol) {
        throw std::invalid_argument("Could not parse SMILES: " + smiles);
      }
      return balaban_j(*mol);
    },
    py::arg("smiles"),
    "Balaban's J connectivity index for one SMILES string. RDKit implements this only in "
    "Python; this is a port of GraphDescriptors.BalabanJ.");
  m.def("_set_sa_score_fragment_path",
        &set_sa_score_fragment_path,
        py::arg("path"),
        "Internal. Points the SA score at the fragment table shipped with the package; "
        "cuik_molmaker calls this on import and the table is read on first use.");
  m.def(
    "sa_score",
    [](const std::string& smiles) {
      const std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
      if (!mol) {
        throw std::invalid_argument("Could not parse SMILES: " + smiles);
      }
      return sa_score(*mol);
    },
    py::arg("smiles"),
    "Synthetic accessibility score for one SMILES string, in [1, 10]. Adopts the "
    "normalization fix from RDKit PR #9501, so values above 8 differ from the Contrib "
    "script, which is discontinuous there. Raises for a molecule with no atoms, for "
    "which the score is undefined.");
  m.def("list_all_molecular_descriptors",
        &list_all_molecular_descriptors,
        "Returns the descriptor names accepted by batch_molecular_descriptors.");
  m.def("batch_molecular_descriptors",
        &batch_molecular_descriptors,
        py::arg("smiles_list"),
        py::arg("descriptor_names"),
        py::arg("num_threads") = 0,
        "Accepts a list of SMILES strings and descriptor names and returns a "
        "(num_molecules, num_descriptors) NumPy array. The batch is split across threads with "
        "the GIL released, using at most the hardware concurrency; SMILES that do not parse give "
        "a row of NaN, and a descriptor that rejects a molecule gives NaN for that cell.");

  m.def(
    "batch_molecular_descriptors_from_binary",
    [](const std::vector<py::bytes>& mol_binaries, const std::vector<std::string>& descriptor_names, int num_threads) {
      std::vector<std::string> blobs;
      blobs.reserve(mol_binaries.size());
      for (const py::bytes& blob : mol_binaries) {
        blobs.emplace_back(blob);
      }
      return batch_molecular_descriptors_from_binary(blobs, descriptor_names, num_threads);
    },
    py::arg("mol_binaries"),
    py::arg("descriptor_names"),
    py::arg("num_threads") = 0,
    "Accepts a list of RDKit binary mol pickles (Mol.ToBinary()) and returns the same array "
    "as batch_molecular_descriptors. Unlike the SMILES entry point this preserves the caller's "
    "molecule exactly, including explicit hydrogens, which BalabanJ, BertzCT and SAScore read.");

  // Reaction featurization (CGR)
  m.def("reaction_mode_to_int",
        &reaction_mode_to_int,
        "Convert a reaction mode name (e.g. 'REAC_DIFF') to its integer enum value; raises on an unknown name.");
  m.def(
    "batch_reaction_featurizer",
    [](const std::vector<std::string>& reac_smiles_list,
       const std::vector<std::string>& prod_smiles_list,
       const py::array_t<int64_t>&     atom_property_list_onehot,
       const py::array_t<int64_t>&     atom_property_list_float,
       const py::array_t<int64_t>&     bond_property_list,
       bool                            keep_h,
       bool                            add_h,
       bool                            offset_carbon,
       int64_t                         mode_int) {
      return batch_reaction_featurizer(reac_smiles_list,
                                       prod_smiles_list,
                                       atom_property_list_onehot,
                                       atom_property_list_float,
                                       bond_property_list,
                                       keep_h,
                                       add_h,
                                       offset_carbon,
                                       ReactionMode(mode_int));
    },
    "Accepts lists of reactant and product SMILES strings and returns a list of NumPy arrays "
    "representing the Condensed Graph of Reaction (CGR) atom and bond features of the reactions. "
    "SMILES must be atom-mapped and providing a correct, unique mapping is the caller's "
    "responsibility (uniqueness is not validated). keep_h keeps explicit hydrogens already in the "
    "SMILES; add_h adds new unmapped hydrogens that become phantom atoms in the CGR.");
}

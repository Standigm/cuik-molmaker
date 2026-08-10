# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""Tests for featurizing molecules supplied as RDKit binary pickles.

The binary entry point exists so that callers holding RWMol objects get features for those
exact objects. A SMILES round-trip preserves chemistry but rewrites atom and bond ordering,
which changes the neighbour-relative chiral tag; these tests pin both properties.
"""

import numpy as np
import pytest
from rdkit import Chem

import cuik_molmaker

ATOM_ONEHOT = ["atomic-number", "total-degree", "formal-charge", "chirality", "num-hydrogens", "hybridization"]
ATOM_FLOAT = ["aromatic", "in-ring"]
BOND_FEATURES = ["bond-type-onehot", "stereo", "conjugated", "in-ring"]

SMILES = [
    "CCO",
    "c1ccccc1",
    "C",
    "C[C@@H](N)C(=O)O",
    "O=C(COc1nnc(SCC(=O)c2ccc3c(c2)OCCO3)n1C1Cc2ccccc2C1)c1ccc(F)cc1",
    "N#Cc1ccc(C=Cc2ccc(C=Cc3ccccc3C#N)cc2)cc1",
]
# Capped BRICS-style fragments: dummy atoms plus stereocentres, the case SMILES cannot serve.
FRAGMENT_SMILES = ["*C(=O)NC[C@H](*)O", "*[C@H]1CC[NH+]1*", "*N1C[C@@H](C)O[C@H](C(F)(F)F)C1"]


@pytest.fixture(scope="module")
def property_arrays():
    return (
        cuik_molmaker.atom_onehot_feature_names_to_array(ATOM_ONEHOT),
        cuik_molmaker.atom_float_feature_names_to_array(ATOM_FLOAT),
        cuik_molmaker.bond_feature_names_to_array(BOND_FEATURES),
    )


def _from_smiles(smiles, arrays):
    onehot, floats, bonds = arrays
    return cuik_molmaker.batch_mol_featurizer(smiles, onehot, floats, bonds, False, False, True, False)


def _from_binary(mols, arrays):
    onehot, floats, bonds = arrays
    return cuik_molmaker.batch_mol_featurizer_from_binary(
        [mol.ToBinary() for mol in mols], onehot, floats, bonds, False, True, False
    )


def test_binary_input_matches_smiles_input(property_arrays):
    """For molecules parsed from SMILES, both entry points must agree exactly."""
    mols = [Chem.MolFromSmiles(smi) for smi in SMILES]
    from_smiles = _from_smiles(SMILES, property_arrays)
    from_binary = _from_binary(mols, property_arrays)

    assert len(from_smiles) == len(from_binary)
    for expected, actual in zip(from_smiles, from_binary):
        np.testing.assert_array_equal(expected, actual)


def test_binary_input_preserves_chiral_tags_of_fragments(property_arrays):
    """A SMILES round-trip may flip CW/CCW; the binary round-trip must not."""
    mols = [Chem.MolFromSmiles(smi) for smi in FRAGMENT_SMILES]
    chirality_offset = sum(
        cuik_molmaker.atom_onehot_feature_names_to_array([name]).size and width
        for name, width in zip(ATOM_ONEHOT[:3], (101, 7, 6))
    )

    atom_features = _from_binary(mols, property_arrays)[0]
    chirality_block = atom_features[:, chirality_offset : chirality_offset + 5]
    decoded = np.argmax(chirality_block, axis=1)

    expected = np.concatenate([[int(a.GetChiralTag()) for a in mol.GetAtoms()] for mol in mols])
    np.testing.assert_array_equal(decoded, np.minimum(expected, 4))


def test_binary_input_preserves_atom_order(property_arrays):
    """Atom rows must follow the molecule's own atom order, not a canonical one."""
    mols = [Chem.MolFromSmiles(smi) for smi in SMILES]
    atom_features = _from_binary(mols, property_arrays)[0]

    # The atomic-number one-hot block is first and 101 wide (1-100 plus unknown).
    atomic_num = np.argmax(atom_features[:, :101], axis=1) + 1
    expected = np.concatenate([[a.GetAtomicNum() for a in mol.GetAtoms()] for mol in mols])
    np.testing.assert_array_equal(atomic_num, expected)


def test_renumbered_molecule_yields_renumbered_rows(property_arrays):
    """Renumbering a molecule must permute the output rows the same way."""
    mol = Chem.MolFromSmiles(SMILES[4])
    order = list(reversed(range(mol.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(mol, order)

    original = _from_binary([mol], property_arrays)[0]
    permuted = _from_binary([renumbered], property_arrays)[0]
    np.testing.assert_array_equal(original[order], permuted)


def test_empty_batch_is_handled(property_arrays):
    atom_features, bond_features, edge_index, _rev, batch = _from_binary([], property_arrays)
    assert atom_features.shape[0] == 0
    assert bond_features.shape[0] == 0
    assert edge_index.shape[1] == 0
    assert batch.shape[0] == 0


def test_invalid_pickle_does_not_crash(property_arrays):
    """A corrupt blob must yield an empty graph rather than terminating the process."""
    onehot, floats, bonds = property_arrays
    good = Chem.MolFromSmiles("CCO")
    arrays = cuik_molmaker.batch_mol_featurizer_from_binary(
        [good.ToBinary(), b"not a molecule"], onehot, floats, bonds, False, True, False
    )
    # The valid molecule still contributes its three atoms; the invalid one contributes none.
    assert arrays[0].shape[0] == good.GetNumAtoms()

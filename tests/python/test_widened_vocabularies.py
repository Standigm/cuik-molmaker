# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth coverage for the two one-hot vocabularies that were widened.

The checked-in reference features predate these blocks, so reference_layout.py excludes them
from that comparison. They are checked here against RDKit instead, which is an independent
oracle rather than a snapshot of this library's own output.
"""

import numpy as np
import pytest
from rdkit import Chem

import cuik_molmaker

# ChemGL's vocabularies, which these now match: bond type is SINGLE/DOUBLE/TRIPLE/AROMATIC
# plus a genuine catch-all, and hydrogen counts span 0-8 before saturating.
BOND_TYPE_ORDER = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
BOND_TYPE_WIDTH = len(BOND_TYPE_ORDER) + 1
NUM_HYDROGENS_MAX = 8
NUM_HYDROGENS_WIDTH = NUM_HYDROGENS_MAX + 2


def _bond_type_onehot(smiles):
    arrays = cuik_molmaker.batch_mol_featurizer(
        [smiles],
        cuik_molmaker.atom_onehot_feature_names_to_array([]),
        cuik_molmaker.atom_float_feature_names_to_array([]),
        cuik_molmaker.bond_feature_names_to_array(["bond-type-onehot"]),
        False,
        False,
        True,
        False,
    )
    return arrays[1]


def _num_hydrogens_onehot(smiles):
    arrays = cuik_molmaker.batch_mol_featurizer(
        [smiles],
        cuik_molmaker.atom_onehot_feature_names_to_array(["num-hydrogens"]),
        cuik_molmaker.atom_float_feature_names_to_array([]),
        cuik_molmaker.bond_feature_names_to_array([]),
        False,
        False,
        True,
        False,
    )
    return arrays[0]


def test_block_widths():
    assert _bond_type_onehot("CCO").shape[1] == BOND_TYPE_WIDTH
    assert _num_hydrogens_onehot("CCO").shape[1] == NUM_HYDROGENS_WIDTH


@pytest.mark.parametrize(
    "smiles",
    ["CCO", "c1ccccc1", "C=C", "C#N", "O=S(=O)([O-])[O-]", "c1cc[nH]c1", "N->[Cu+2]", "C~C"],
)
def test_bond_type_matches_rdkit(smiles):
    """Every bond decodes to its own RDKit type, or to the catch-all if it has no slot."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    onehot = _bond_type_onehot(smiles)

    expected = []
    for bond in mol.GetBonds():
        index = BOND_TYPE_ORDER.index(bond.GetBondType()) if bond.GetBondType() in BOND_TYPE_ORDER else len(BOND_TYPE_ORDER)
        expected.extend([index, index])  # duplicate_edges=True stores both directions

    assert onehot.shape[0] == len(expected)
    np.testing.assert_array_equal(np.argmax(onehot, axis=1), expected)
    np.testing.assert_array_equal(onehot.sum(axis=1), np.ones(len(expected)))


def test_aromatic_bonds_are_distinct_from_the_catch_all():
    """The regression this widening fixes: AROMATIC used to share a slot with everything else."""
    aromatic = np.argmax(_bond_type_onehot("c1ccccc1"), axis=1)
    dative = np.argmax(_bond_type_onehot("N->[Cu+2]"), axis=1)
    unspecified = np.argmax(_bond_type_onehot("C~C"), axis=1)

    assert set(aromatic) == {BOND_TYPE_ORDER.index(Chem.BondType.AROMATIC)}
    assert set(dative) == {len(BOND_TYPE_ORDER)}
    assert set(unspecified) == {len(BOND_TYPE_ORDER)}
    assert set(aromatic).isdisjoint(set(dative) | set(unspecified))


@pytest.mark.parametrize("smiles", ["C", "CCO", "N", "O", "[XeH6]", "[SiH4]", "c1cc[nH]c1", "[NH4+]"])
def test_num_hydrogens_matches_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    onehot = _num_hydrogens_onehot(smiles)

    expected = [
        min(atom.GetTotalNumHs(), NUM_HYDROGENS_MAX + 1) for atom in mol.GetAtoms()
    ]
    np.testing.assert_array_equal(np.argmax(onehot, axis=1), expected)
    np.testing.assert_array_equal(onehot.sum(axis=1), np.ones(len(expected)))


def test_hydrogen_counts_above_five_are_distinguished():
    """The regression this widening fixes: counts used to saturate at 5."""
    xenon = _num_hydrogens_onehot("[XeH6]")
    assert int(np.argmax(xenon, axis=1)[0]) == 6

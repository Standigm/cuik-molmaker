# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import lzma
import os
import pickle

import numpy as np
import pytest

import cuik_molmaker
from reference_layout import assert_matches_reference

REACTION_MODES = [
    "REAC_DIFF",
    "REAC_PROD",
    "PROD_DIFF",
    "REAC_DIFF_BALANCE",
    "REAC_PROD_BALANCE",
    "PROD_DIFF_BALANCE",
]

# V1/V2/ORGANIC share the same bond features (bond_fdim=14 each, 28 total in CGR).
# RIGR uses reduced atom AND bond features (bond_fdim=2 each, 4 total in CGR).
FEATURIZER_CONFIGS = {
    "V1": {
        "atom_onehot": [
            "atomic-number",
            "total-degree",
            "formal-charge",
            "chirality",
            "num-hydrogens",
            "hybridization",
        ],
        "atom_float": ["aromatic", "mass"],
        "bond": ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"],
    },
    "V2": {
        "atom_onehot": [
            "atomic-number-common",
            "total-degree",
            "formal-charge",
            "chirality",
            "num-hydrogens",
            "hybridization-expanded",
        ],
        "atom_float": ["aromatic", "mass"],
        "bond": ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"],
    },
    "ORGANIC": {
        "atom_onehot": [
            "atomic-number-organic",
            "total-degree",
            "formal-charge",
            "chirality",
            "num-hydrogens",
            "hybridization-organic",
        ],
        "atom_float": ["aromatic", "mass"],
        "bond": ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"],
    },
    "RIGR": {
        "atom_onehot": ["atomic-number-common", "total-degree", "num-hydrogens"],
        "atom_float": ["mass"],
        "bond": ["is-null", "in-ring"],  # RIGR reduces bond features too
    },
}


@pytest.mark.parametrize("atom_featurizer_version", list(FEATURIZER_CONFIGS.keys()))
@pytest.mark.parametrize("reaction_mode", REACTION_MODES)
def test_batch_reaction_featurizer(
    test_data_path, atom_featurizer_version, reaction_mode
):
    cfg = FEATURIZER_CONFIGS[atom_featurizer_version]
    atom_onehot = cuik_molmaker.atom_onehot_feature_names_to_array(cfg["atom_onehot"])
    atom_float = cuik_molmaker.atom_float_feature_names_to_array(cfg["atom_float"])
    bond_feats = cuik_molmaker.bond_feature_names_to_array(cfg["bond"])
    mode_int = cuik_molmaker.reaction_mode_to_int(reaction_mode)

    ref_file = f"sample_rxns_100_{atom_featurizer_version}_{reaction_mode}_ref.xz"
    ref_path = os.path.join(test_data_path, ref_file)
    with lzma.open(ref_path, "rb") as f:
        ref = pickle.load(f)

    V, E, edge_index, rev_edge_index, batch = cuik_molmaker.batch_reaction_featurizer(
        ref["reac_smiles"],
        ref["prod_smiles"],
        atom_onehot,
        atom_float,
        bond_feats,
        True,  # keep_h — required for atom-mapped reactions with explicit H
        False,  # add_h
        False,  # offset_carbon
        mode_int,
    )

    assert_matches_reference(
        V,
        ref["V"],
        # CGR concatenates the full atom block with a second copy whose atomic-number
        # one-hot has been stripped.
        [
            (cfg["atom_onehot"], len(cfg["atom_float"])),
            (cfg["atom_onehot"][1:], len(cfg["atom_float"])),
        ],
        "atom_onehot",
        err_msg=f"[{atom_featurizer_version}/{reaction_mode}] atom feats mismatch "
        "outside the widened blocks",
    )
    assert_matches_reference(
        E,
        ref["E"],
        [(cfg["bond"], 0), (cfg["bond"], 0)],
        "bond",
        err_msg=f"[{atom_featurizer_version}/{reaction_mode}] bond feats mismatch "
        "outside the widened blocks",
    )
    np.testing.assert_allclose(
        ref["edge_index"],
        edge_index,
        err_msg=f"[{atom_featurizer_version}/{reaction_mode}] edge_index mismatch",
    )
    np.testing.assert_allclose(
        ref["rev_edge_index"],
        rev_edge_index,
        err_msg=f"[{atom_featurizer_version}/{reaction_mode}] rev_edge_index mismatch",
    )
    np.testing.assert_allclose(
        ref["batch"],
        batch,
        err_msg=f"[{atom_featurizer_version}/{reaction_mode}] batch mismatch",
    )

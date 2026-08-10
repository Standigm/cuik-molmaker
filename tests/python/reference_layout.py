# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""Compare against the checked-in reference features across a vocabulary change.

Two one-hot vocabularies were widened so that they no longer conflate distinct values:
``bond-type-onehot`` gained an explicit AROMATIC slot, freeing its last slot to mean
"anything else", and ``num-hydrogens`` now spans 0-8 rather than saturating at 5. The
reference pickles predate both, so those blocks are the only ones that can no longer be
compared. Everything else still is, which keeps the references meaningful as an independent
oracle rather than something regenerated from the code under test.

Block widths are measured from the library rather than tabulated, so this keeps working if a
vocabulary changes again; only the historical widths have to be recorded here. Callers
describe the column layout as one entry per component, which covers both plain molecule
features and Condensed Graph of Reaction features.

The widened blocks themselves are covered against RDKit directly in
test_widened_vocabularies.py, so nothing is left unverified.
"""

import numpy as np

import cuik_molmaker

# Width these features had when the reference files were generated.
REFERENCE_WIDTHS = {"num-hydrogens": 6, "bond-type-onehot": 4}

_PROBE_SMILES = "CCO"
_EMPTY = None


def _empty_arrays():
    global _EMPTY
    if _EMPTY is None:
        _EMPTY = (
            cuik_molmaker.atom_onehot_feature_names_to_array([]),
            cuik_molmaker.atom_float_feature_names_to_array([]),
            cuik_molmaker.bond_feature_names_to_array([]),
        )
    return _EMPTY


def _measure_width(name, kind):
    """Return one feature's current column width, measured in isolation."""
    empty_atom_onehot, empty_atom_float, empty_bond = _empty_arrays()
    if kind == "atom_onehot":
        arrays = cuik_molmaker.batch_mol_featurizer(
            [_PROBE_SMILES],
            cuik_molmaker.atom_onehot_feature_names_to_array([name]),
            empty_atom_float,
            empty_bond,
            False,
            False,
            True,
            False,
        )
        return arrays[0].shape[1]
    if kind == "bond":
        arrays = cuik_molmaker.batch_mol_featurizer(
            [_PROBE_SMILES],
            empty_atom_onehot,
            empty_atom_float,
            cuik_molmaker.bond_feature_names_to_array([name]),
            False,
            False,
            True,
            False,
        )
        return arrays[1].shape[1]
    raise ValueError(f"unknown kind: {kind}")


def _retained_columns(layouts, kind, use_reference_widths):
    """Column indices to compare, skipping blocks whose vocabulary changed."""
    keep, offset = [], 0
    for names, num_float in layouts:
        widths = [_measure_width(name, kind) for name in names]
        if use_reference_widths:
            widths = [REFERENCE_WIDTHS.get(name, width) for name, width in zip(names, widths)]
        for name, width in zip(names, widths):
            if name not in REFERENCE_WIDTHS:
                keep.extend(range(offset, offset + width))
            offset += width
        keep.extend(range(offset, offset + num_float))  # float features are unchanged
        offset += num_float
    return np.asarray(keep, dtype=np.int64), offset


def assert_matches_reference(actual, reference, layouts, kind, err_msg=""):
    """Assert ``actual`` matches ``reference`` on every column whose vocabulary is unchanged.

    Parameters
    ----------
    actual, reference
        Feature matrices from the current build and from the checked-in reference file.
    layouts
        One ``(feature_names, num_float_columns)`` pair per component, in column order. Plain
        molecule features have a single component; Condensed Graph of Reaction features have
        two, the second with the atomic-number block stripped.
    kind
        Either ``"atom_onehot"`` or ``"bond"``.
    """
    actual_columns, actual_width = _retained_columns(layouts, kind, False)
    reference_columns, reference_width = _retained_columns(layouts, kind, True)
    assert actual_width == actual.shape[1], (
        f"described layout is {actual_width} columns but the array has {actual.shape[1]}"
    )
    assert reference_width == reference.shape[1], (
        f"described reference layout is {reference_width} columns but the file has "
        f"{reference.shape[1]}; REFERENCE_WIDTHS is probably stale"
    )
    np.testing.assert_allclose(
        np.asarray(reference)[:, reference_columns],
        np.asarray(actual)[:, actual_columns],
        err_msg=err_msg,
    )

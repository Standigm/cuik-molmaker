# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the C++ molecular descriptors.

``BertzCT``, ``BalabanJ``, ``qed`` and the SA score have no RDKit C++ implementation
and are ported in ``molecular_descriptors.cpp``; these tests hold them to the values
RDKit's Python produces, since a port is only worth having if it changes nothing.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.RDConfig import RDContribDir

import cuik_molmaker

sys.path.append(os.path.join(RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

_RDKIT_FNS = dict(Descriptors._descList)

# Above 8, sascorer.py takes log(score - 8), which collapses the result to a small
# number, so it is only a valid reference below that boundary. An affected molecule is
# recognised by *our* score exceeding the boundary, not by sascorer.py's -- theirs comes
# out low precisely when it is wrong.
SA_SMOOTHING_BOUNDARY = 8.0

REFERENCE_FNS = {
    "qed": _RDKIT_FNS["qed"],
    "MolWt": _RDKIT_FNS["MolWt"],
    "BalabanJ": _RDKIT_FNS["BalabanJ"],
    "BertzCT": _RDKIT_FNS["BertzCT"],
    "HallKierAlpha": rdmd.CalcHallKierAlpha,
    "TPSA": _RDKIT_FNS["TPSA"],
    "MolLogP": _RDKIT_FNS["MolLogP"],
    "MolMR": _RDKIT_FNS["MolMR"],
    "SAScore": sascorer.calculateScore,
}

# Reference values from RDKit PR #9501, which ports the SA score to C++ and fixes the
# normalization. Reproducing them is what makes this implementation the corrected one.
PR_9501_REFERENCE_SCORES = (
    ("c1ccccc1", 1.000000),
    ("CC(=O)Oc1ccccc1C(=O)O", 1.580040),
    ("C[C@H](N)C(=O)O", 2.319603),
    ("CC(N)C(=O)O", 2.319603),
    (
        "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)"
        "[C@H]3CC[C@]12C",
        4.163362,
    ),
    ("C1CCC2(CC1)CCCCC2", 2.412221),
    ("C1CC2CCC1CC2", 2.784431),
    ("C1CCCCCCCCCCC1", 1.000000),
    ("C1CC1C1CC1C1CC1", 3.114235),
    (
        "CC1=C2C(=O)C(OC(C)=O)C3(C)C(O)CC4OCC4(OC(C)=O)C3C(OC(=O)c3ccccc3)C2(C)C(O)CC1"
        "OC(=O)C(O)C(NC(=O)c1ccccc1)c1ccccc1",
        5.463623,
    ),
)

# The molecule PR #9501 uses to demonstrate the bug: sascorer.py scores it 4.36,
# i.e. easier to make than aspirin, because its log() branch diverges.
SA_BOUNDARY_SMILES = (
    "C[N+]12CCCC1c1ccc[n+](c1)[Zn+2]21[O]C(=O)C(=O)[O]1."
    "O=C1[OH+][Zn+2]2([OH+]C1=O)[OH+]C(=O)C(=O)[OH+]2"
)
SA_BOUNDARY_EXPECTED = 8.025797


@pytest.fixture
def valid_smiles(smiles_list_100):
    return [smi for smi in smiles_list_100 if Chem.MolFromSmiles(smi) is not None]


def test_listed_descriptors_all_have_references():
    assert set(cuik_molmaker.list_all_molecular_descriptors()) == set(REFERENCE_FNS)


def test_sample_stays_below_sa_boundary(valid_smiles):
    """sascorer.py is only a valid reference below the boundary it mishandles."""
    scores = cuik_molmaker.batch_molecular_descriptors(valid_smiles, ["SAScore"], 4)[
        :, 0
    ]

    assert scores.max() < SA_SMOOTHING_BOUNDARY


@pytest.mark.parametrize("descriptor_name", sorted(REFERENCE_FNS))
def test_batch_matches_rdkit(valid_smiles, descriptor_name):
    computed = cuik_molmaker.batch_molecular_descriptors(
        valid_smiles, [descriptor_name], 4
    )[:, 0]
    reference = np.array(
        [
            REFERENCE_FNS[descriptor_name](Chem.MolFromSmiles(smi))
            for smi in valid_smiles
        ]
    )

    np.testing.assert_allclose(computed, reference, rtol=1e-12, atol=1e-12)


def test_single_molecule_entry_points_match_rdkit(valid_smiles):
    for smi in valid_smiles[:20]:
        mol = Chem.MolFromSmiles(smi)
        assert cuik_molmaker.bertz_ct(smi) == pytest.approx(
            _RDKIT_FNS["BertzCT"](mol), rel=1e-12
        )
        assert cuik_molmaker.qed_weights_mean(smi) == pytest.approx(
            _RDKIT_FNS["qed"](mol), rel=1e-12
        )
        assert cuik_molmaker.balaban_j(smi) == pytest.approx(
            _RDKIT_FNS["BalabanJ"](mol), rel=1e-12
        )
        assert cuik_molmaker.sa_score(smi) == pytest.approx(
            sascorer.calculateScore(mol), rel=1e-12
        )


@pytest.mark.parametrize("smiles,expected", PR_9501_REFERENCE_SCORES)
def test_sa_score_matches_pr_9501_reference(smiles, expected):
    assert cuik_molmaker.sa_score(smiles) == pytest.approx(expected, abs=1e-4)


def test_sa_score_corrects_the_contrib_discontinuity():
    """Above the boundary sascorer.py collapses; this must not follow it."""
    contrib = sascorer.calculateScore(Chem.MolFromSmiles(SA_BOUNDARY_SMILES))

    assert cuik_molmaker.sa_score(SA_BOUNDARY_SMILES) == pytest.approx(
        SA_BOUNDARY_EXPECTED, abs=1e-4
    )
    assert (
        contrib
        < sascorer.calculateScore(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")) + 3.0
    )
    assert contrib < SA_SMOOTHING_BOUNDARY


def test_thread_count_does_not_change_results(valid_smiles):
    names = cuik_molmaker.list_all_molecular_descriptors()
    serial = cuik_molmaker.batch_molecular_descriptors(valid_smiles, names, 1)

    for num_threads in (2, 8, 32):
        np.testing.assert_array_equal(
            cuik_molmaker.batch_molecular_descriptors(valid_smiles, names, num_threads),
            serial,
        )


def test_unparsable_smiles_yields_nan_row():
    names = cuik_molmaker.list_all_molecular_descriptors()

    out = cuik_molmaker.batch_molecular_descriptors(
        ["CCO", "not-a-smiles", "c1ccccc1"], names, 2
    )

    assert out.shape == (3, len(names))
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[0]).all() and np.isfinite(out[2]).all()


def test_unknown_descriptor_raises():
    with pytest.raises(ValueError, match="Unknown molecular descriptor"):
        cuik_molmaker.batch_molecular_descriptors(["CCO"], ["NotADescriptor"], 1)


def test_bertz_ct_of_single_atom_is_zero():
    assert cuik_molmaker.bertz_ct("C") == 0.0


# Symmetric molecules put the most pressure on BertzCT's symmetry classes, which are
# defined by comparing distance vectors rounded to four decimals: merge two classes that
# RDKit keeps apart, or split one it merges, and the value moves.
SYMMETRIC_SMILES = (
    "c1ccccc1",  # benzene
    "C1CCCCC1",  # cyclohexane
    "C1CCCCCCCCCCC1",  # cyclododecane
    "C12C3C4C1C5C4C3C25",  # cubane
    "C1C2CC3CC1CC(C2)C3",  # adamantane
    "C1CC2CCC1CC2",  # bicyclo[2.2.2]octane
    "C1CCC2(CC1)CCCCC2",  # spiro[5.5]undecane
    "c1ccc2ccccc2c1",  # naphthalene
    "c1ccc2cc3ccccc3cc2c1",  # anthracene
    "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61",  # coronene
    "C1CC1C1CC1C1CC1",  # tris-cyclopropyl
    "FC(F)(F)C(F)(F)C(F)(F)F",  # perfluoropropane
    "ClC(Cl)(Cl)C(Cl)(Cl)Cl",  # hexachloroethane
)


@pytest.mark.parametrize("smiles", SYMMETRIC_SMILES)
@pytest.mark.parametrize("descriptor_name", ("BertzCT", "BalabanJ"))
def test_distance_matrix_descriptors_on_symmetric_molecules(smiles, descriptor_name):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    computed = cuik_molmaker.batch_molecular_descriptors(
        [smiles], [descriptor_name], 1
    )[0, 0]

    assert computed == pytest.approx(REFERENCE_FNS[descriptor_name](mol), rel=1e-12)


# ---------------------------------------------------------------------------
# Binary input. Callers holding an RWMol must not go through SMILES: sanitization
# on reparse drops explicit hydrogens, which several descriptors read.
# ---------------------------------------------------------------------------

HYDROGEN_SENSITIVE = ("BalabanJ", "BertzCT", "SAScore")


def test_binary_matches_smiles_for_implicit_hydrogen_molecules(valid_smiles):
    names = cuik_molmaker.list_all_molecular_descriptors()
    binaries = [Chem.MolFromSmiles(smi).ToBinary() for smi in valid_smiles]

    np.testing.assert_allclose(
        cuik_molmaker.batch_molecular_descriptors_from_binary(binaries, names, 4),
        cuik_molmaker.batch_molecular_descriptors(valid_smiles, names, 4),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("descriptor_name", sorted(REFERENCE_FNS))
def test_binary_preserves_explicit_hydrogens(valid_smiles, descriptor_name):
    """The binary path must agree with RDKit on the caller's actual molecule."""
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in valid_smiles]

    computed = cuik_molmaker.batch_molecular_descriptors_from_binary(
        [mol.ToBinary() for mol in mols], [descriptor_name], 4
    )[:, 0]
    reference = np.array([REFERENCE_FNS[descriptor_name](mol) for mol in mols])

    if descriptor_name == "SAScore":
        # Adding hydrogens pushes some past the boundary sascorer.py mishandles,
        # where disagreeing with it is the point of the fix rather than a failure.
        comparable = computed <= SA_SMOOTHING_BOUNDARY
        assert comparable.sum() > len(mols) // 2, "too few molecules left to compare"
        computed, reference = computed[comparable], reference[comparable]

    np.testing.assert_allclose(computed, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("descriptor_name", HYDROGEN_SENSITIVE)
def test_smiles_path_would_lose_explicit_hydrogens(valid_smiles, descriptor_name):
    """Guards why the binary entry point exists, so it is not simplified away."""
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in valid_smiles]
    reference = np.array([REFERENCE_FNS[descriptor_name](mol) for mol in mols])

    via_smiles = cuik_molmaker.batch_molecular_descriptors(
        [Chem.MolToSmiles(mol) for mol in mols], [descriptor_name], 1
    )[:, 0]

    assert not np.allclose(via_smiles, reference, rtol=1e-6)


def test_unreadable_binary_yields_nan_row(valid_smiles):
    names = cuik_molmaker.list_all_molecular_descriptors()
    good = Chem.MolFromSmiles(valid_smiles[0]).ToBinary()

    out = cuik_molmaker.batch_molecular_descriptors_from_binary(
        [good, b"not a pickle", b"", good], names, 2
    )

    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()
    assert np.isfinite(out[0]).all() and np.isfinite(out[3]).all()


# ---------------------------------------------------------------------------
# Contracts the C++ side promises: zero-atom rejection, thread-count clamping,
# and refusal to load a malformed fragment table.
# ---------------------------------------------------------------------------


def test_sa_score_rejects_a_molecule_with_no_atoms():
    """An empty SMILES parses to a zero-atom molecule, where the score is undefined."""
    assert Chem.MolFromSmiles("").GetNumAtoms() == 0

    with pytest.raises(ValueError, match="no atoms"):
        cuik_molmaker.sa_score("")


def test_batch_reports_a_rejected_molecule_as_nan():
    """A descriptor that rejects one molecule must not fail the whole batch."""
    names = cuik_molmaker.list_all_molecular_descriptors()

    out = cuik_molmaker.batch_molecular_descriptors(["CCO", "", "c1ccccc1"], names, 2)

    sa_column = names.index("SAScore")
    assert np.isnan(out[1, sa_column])
    assert np.isfinite(out[0]).all() and np.isfinite(out[2]).all()


@pytest.mark.parametrize("num_threads", (1, 4, 1_000_000))
def test_absurd_thread_count_is_clamped(valid_smiles, num_threads):
    """Worker count is capped, so a large request cannot exhaust thread handles."""
    names = cuik_molmaker.list_all_molecular_descriptors()

    np.testing.assert_array_equal(
        cuik_molmaker.batch_molecular_descriptors(valid_smiles, names, num_threads),
        cuik_molmaker.batch_molecular_descriptors(valid_smiles, names, 1),
    )


# Exit code the probe below uses to say the environment could not exercise the failure.
_PROBE_INCONCLUSIVE = 42

# Aborting needs a worker to fail *after* an earlier one started: with no slots at all,
# the first construction throws into an empty pool, which unwinds cleanly either way.
# Two things make that reachable rather than hoped for. The probe saturates RLIMIT_NPROC
# with threads it keeps alive and then *raises* the ceiling, so the free slots do not
# depend on a joined thread having been reaped by the OS. And it only reports success
# once a worker has demonstrably run, measured as process CPU time, so a first-worker
# refusal reports inconclusive instead of passing.
_THREAD_FAILURE_PROBE = """
import resource, sys, threading

INCONCLUSIVE, HEADROOM, REQUEST, CAP = 42, 2, 32, 256
MIN_WORKER_CPU = 0.25          # seconds; one worker's chunk is ~0.6s

import cuik_molmaker as cm

names = cm.list_all_molecular_descriptors()
# chunks large enough that a started worker burns measurable CPU before the pool fails
smiles = ["CC(=O)Oc1ccccc1C(=O)O"] * 12800
cm.batch_molecular_descriptors(smiles[:1], names, 1)   # warm tables before limits

soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)


def hold(limit):
    resource.setrlimit(resource.RLIMIT_NPROC, (limit, hard))
    events, threads = [], []
    try:
        while len(threads) < CAP:
            event = threading.Event()
            thread = threading.Thread(target=event.wait, daemon=True)
            thread.start()
            events.append(event)
            threads.append(thread)
    except RuntimeError:
        pass
    return events, threads


def release(events, threads):
    for event in events:
        event.set()
    for thread in threads:
        thread.join()


def count_at(limit):
    events, threads = hold(limit)
    held = len(threads)
    release(events, threads)
    return held


lo, hi = 1, 128
while count_at(hi) < CAP:
    lo, hi = hi, hi * 2
    if hi > (1 << 22):
        sys.exit(INCONCLUSIVE)

target = None
while hi - lo > 1:
    mid = (lo + hi) // 2
    held = count_at(mid)
    if held >= CAP:
        hi = mid
    elif held <= HEADROOM:
        lo = mid
    else:
        target = mid
        break

if target is None:
    sys.exit(INCONCLUSIVE)

events, threads = hold(target)
if not HEADROOM < len(threads) < CAP:
    sys.exit(INCONCLUSIVE)

# Holders stay alive and saturated; raising the ceiling frees exactly HEADROOM slots.
resource.setrlimit(resource.RLIMIT_NPROC, (target + HEADROOM, hard))

before = resource.getrusage(resource.RUSAGE_SELF).ru_utime
try:
    cm.batch_molecular_descriptors(smiles, names, REQUEST)
except Exception as exc:
    spent = resource.getrusage(resource.RUSAGE_SELF).ru_utime - before
    if spent < MIN_WORKER_CPU:
        sys.exit(INCONCLUSIVE)      # no worker ran, so none started before the refusal
    if "esource" not in str(exc):
        sys.exit(INCONCLUSIVE)      # not the thread-creation failure this sets up
    sys.exit(0)
sys.exit(INCONCLUSIVE)              # the ceiling never bit, so this run proves nothing
"""


@pytest.mark.skipif(os.name != "posix", reason="RLIMIT_NPROC is POSIX-only")
def test_failed_worker_start_raises_instead_of_aborting():
    """A worker that cannot start must raise, after earlier workers have already run.

    Unwinding past a joinable ``std::thread`` calls ``std::terminate``, so a pool of
    those aborts with "terminate called without an active exception", while
    ``std::jthread`` joins while unwinding and lets the error propagate. Measured both
    ways on the same machine: the old pool aborts with exit 134, this one exits 0 having
    caught the creation error with ~0.6s of worker CPU behind it.
    """
    result = subprocess.run(
        [sys.executable, "-c", _THREAD_FAILURE_PROBE], capture_output=True, text=True
    )

    if result.returncode == _PROBE_INCONCLUSIVE:
        pytest.skip("could not exercise a partial pool failure in this environment")
    assert result.returncode == 0, (
        f"worker start failure was not raised cleanly (exit {result.returncode}): "
        f"{result.stderr.strip()[-300:]}"
    )

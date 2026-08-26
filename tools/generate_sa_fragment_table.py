# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""Convert RDKit's SA_Score fragment pickle into a binary table the C++ code can mmap.

The Contrib pickle costs roughly 90 MB of Python dict and a quarter second to load.
The same 705k Morgan-bit scores fit in ~4 MB here because only a few thousand distinct
scores occur, so each entry stores an index into a small score table rather than a
double. Keys are sorted so lookup is a binary search.

Regenerate when bumping RDKit::

    python tools/generate_sa_fragment_table.py data/sa_score_fragments.bin
"""

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
from rdkit.RDConfig import RDContribDir

MAGIC = b"CUIKSAS1"


def build_table(pickle_path: Path) -> bytes:
    """Serialize the fragment scores into the binary layout the C++ loader expects.

    Parameters
    ----------
    pickle_path : Path
        RDKit's ``fpscores.pkl.gz``. Each row is a score followed by the Morgan bit
        ids that share it.

    Returns
    -------
    bytes
        Magic, entry count, score count, sorted uint32 keys, uint16 score indices,
        then the float64 score table.
    """
    with gzip.open(pickle_path, "rb") as handle:
        rows = pickle.load(handle)

    keys: list[int] = []
    scores: list[float] = []
    for row in rows:
        score = float(row[0])
        for key in row[1:]:
            keys.append(key)
            scores.append(score)

    key_array = np.array(keys, dtype=np.uint64)
    if key_array.max() >= 2**32:
        raise ValueError("Morgan bit ids no longer fit in uint32; widen the format")

    key_array = key_array.astype(np.uint32)
    score_array = np.array(scores, dtype=np.float64)

    order = np.argsort(key_array, kind="stable")
    key_array = key_array[order]
    score_array = score_array[order]
    if len(np.unique(key_array)) != len(key_array):
        raise ValueError("duplicate Morgan bit ids; lookup assumes unique keys")

    score_table = np.unique(score_array)
    if len(score_table) >= 2**16:
        raise ValueError("too many distinct scores for a uint16 index; widen it")
    score_index = np.searchsorted(score_table, score_array).astype(np.uint16)

    return b"".join(
        (
            MAGIC,
            np.array([len(key_array), len(score_table)], dtype=np.uint64).tobytes(),
            key_array.tobytes(),
            score_index.tobytes(),
            score_table.tobytes(),
        )
    )


def main() -> None:
    """Write the binary table to the requested path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--fpscores",
        type=Path,
        default=Path(RDContribDir) / "SA_Score" / "fpscores.pkl.gz",
    )
    args = parser.parse_args()

    payload = build_table(args.fpscores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote {args.output} ({len(payload) / 1e6:.2f} MB) from {args.fpscores}")


if __name__ == "__main__":
    main()

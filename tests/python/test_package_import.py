# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""Tests for what importing the package costs and exposes.

Parallel callers import the package once per worker process for the compiled entry
points alone, so the pandas/scipy-backed helpers must not load until something asks for
them.
"""

import subprocess
import sys

import pytest

import cuik_molmaker

HEAVY_MODULES = ("pandas", "scipy.stats")


def test_import_defers_the_python_helpers(tmp_path):
    # A fresh interpreter, outside the checkout so its build directory cannot shadow the
    # installed package; the parent test process has long since imported both modules.
    probe = (
        "import sys, cuik_molmaker\n"
        "cuik_molmaker.batch_molecular_descriptors(['CCO'], ['MolWt'], 1)\n"
        f"print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout.strip()
    assert loaded == "[]"


def test_lazy_exports_resolve_and_are_listed():
    for name in cuik_molmaker.__all__:
        assert callable(getattr(cuik_molmaker, name))
        assert name in dir(cuik_molmaker)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_attribute"):
        cuik_molmaker.no_such_attribute

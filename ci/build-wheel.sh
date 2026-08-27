#!/bin/bash
# Build, audit and test one wheel. Runs inside a manylinux container, from CI and by hand
# alike, so what a maintainer verifies locally is what the release job publishes.
#
#   docker run --rm -v "$PWD:/src:ro" -v "$PWD/dist:/out" \
#     quay.io/pypa/manylinux_2_28_x86_64 bash /src/ci/build-wheel.sh cp313
#
# The container supplies the glibc baseline: its toolchain targets 2.28, so the artifact
# cannot acquire a symbol version newer than the platform tag promises. Headers come from
# conda-forge because PyPI publishes no rdkit sdist and rdkit-headers trails the library
# releases; headers are text and carry no ABI, so the floor stays the container's.
set -euo pipefail

PY_TAG=$1                                   # cp311 | cp312 | cp313
RDKIT_VERSION=${RDKIT_VERSION:-2026.03.4}
SRC=${SRC:-/src}
OUT=${OUT:-/out}
PYBIN=/opt/python/${PY_TAG}-${PY_TAG}/bin

# A release tag becomes the wheel's version: v0.3.0-standigm.4 -> 0.3.0+standigm.4, the
# PEP 440 local version, which sorts above plain 0.3.0 and names the fork in `pip show`.
# Without it every Standigm tag republishes 0.3.0 and pip, seeing a pinned direct
# reference already satisfied, silently keeps whichever wheel is installed.
if [ -z "${CUIK_MOLMAKER_VERSION:-}" ] && [ "${GITHUB_REF_TYPE:-}" = tag ]; then
  CUIK_MOLMAKER_VERSION=$(printf '%s' "${GITHUB_REF_NAME#v}" | sed 's/-/+/')
fi
export CUIK_MOLMAKER_VERSION="${CUIK_MOLMAKER_VERSION:-}"

echo "::group::${PY_TAG}: dependencies"
if [ ! -d /opt/hdr ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba
  /usr/local/bin/micromamba create -y -q -p /opt/hdr -c conda-forge \
    "librdkit-dev=${RDKIT_VERSION}" libboost-devel >/dev/null
fi
"$PYBIN/pip" install -q "rdkit==${RDKIT_VERSION}" numpy setuptools wheel cmake pybind11 auditwheel
echo "::endgroup::"

PURELIB=$("$PYBIN/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export CUIKMOLMAKER_BUILD_AGAINST_PIP_RDKIT=1
export CUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR="${PURELIB}/rdkit.libs"
export CUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR=/opt/hdr/include/rdkit
export CUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR=/opt/hdr/include
export RDKIT_VERSION
export CUIKMOLMAKER_CXX11_ABI=ON
export PATH="$PYBIN:$PATH"

echo "::group::${PY_TAG}: build"
# A fresh tree per interpreter, so no artifact from a previous tag is picked up.
rm -rf /work && cp -a "$SRC" /work
rm -rf /work/build /work/cuik_molmaker/lib /work/cuik_molmaker/*.so
"$PYBIN/pip" wheel --no-build-isolation --no-deps -w "$OUT/raw" /work
raw=$(ls "$OUT"/raw/cuik_molmaker-*-${PY_TAG}-*.whl | tail -1)
echo "::endgroup::"

echo "::group::${PY_TAG}: platform tag"
# RDKit is excluded rather than vendored: the target environment already has it as the
# rdkit wheel, and bundling a second copy is exactly what linking that wheel avoids.
"$PYBIN/python" -m auditwheel repair --exclude 'libRDKit*' \
  --plat manylinux_2_28_x86_64 --only-plat -w "$OUT" "$raw"
final=$(ls "$OUT"/cuik_molmaker-*-${PY_TAG}-manylinux*.whl | tail -1)
echo "::endgroup::"

echo "${PY_TAG}: auditing $(basename "$final")"
"$PYBIN/python" - "$final" <<'PY'
import pathlib, re, subprocess, sys, tempfile, zipfile

BASELINE = (2, 28)
wheel = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory() as unpacked:
    zipfile.ZipFile(wheel).extractall(unpacked)
    objects = sorted(pathlib.Path(unpacked).rglob("*.so*"))
    assert objects, "no native objects in the wheel"

    worst = (0, 0)
    for so in objects:
        symbols = subprocess.run(["objdump", "-T", str(so)], capture_output=True, text=True).stdout
        worst = max(worst, *(tuple(map(int, v.split("."))) for v in re.findall(r"GLIBC_(\d+\.\d+)", symbols)))
        needed = re.findall(r"Shared library: \[([^\]]+)\]", subprocess.run(
            ["readelf", "-d", str(so)], capture_output=True, text=True).stdout)
        print(f"  {so.name} -> {', '.join(needed)}")

    print(f"  max GLIBC required: {'.'.join(map(str, worst))}")
    assert worst <= BASELINE, (
        f"{wheel.name} requires GLIBC {'.'.join(map(str, worst))}, above the "
        f"{'.'.join(map(str, BASELINE))} its platform tag promises"
    )
PY

echo "::group::${PY_TAG}: installed-wheel checks"
# A plain virtual environment holding the RDKit wheel and this one, and nothing else: no
# compiler, no RDKit headers, no Boost, no conda prefix. That is the property that lets
# consumers install without a toolchain, so the build fails here if it stops holding.
rm -rf "/tmp/probe-${PY_TAG}"
"$PYBIN/python" -m venv "/tmp/probe-${PY_TAG}"
"/tmp/probe-${PY_TAG}/bin/pip" install -q "rdkit==${RDKIT_VERSION}" numpy pandas scipy pytest "$final"

# Run from /tmp so the checkout's own cuik_molmaker directory, which the build writes
# artifacts into, cannot shadow the installed package.
cd /tmp
"/tmp/probe-${PY_TAG}/bin/python" - <<'PY'
from rdkit import Chem
import cuik_molmaker

onehot = cuik_molmaker.atom_onehot_feature_names_to_array(["chirality"])
floats = cuik_molmaker.atom_float_feature_names_to_array(["atomic-number"])
bonds = cuik_molmaker.bond_feature_names_to_array(["bond-type-onehot"])
atoms, edges, *_ = cuik_molmaker.batch_mol_featurizer_from_binary(
    [Chem.MolFromSmiles("c1ccccc1O").ToBinary()], onehot, floats, bonds, False, True, False
)
assert atoms.shape[0] == 7 and edges.shape[0] == 14, (atoms.shape, edges.shape)
print("  featurizes in a prefix with no toolchain")
PY
"/tmp/probe-${PY_TAG}/bin/python" -m pytest /work/tests/python -q
echo "::endgroup::"

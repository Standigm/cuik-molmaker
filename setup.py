# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import configparser
import glob
import os
import platform
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Version for the cuik_molmaker package (cuik_molmaker_pin uses RDKIT_VERSION instead).
CUIK_MOLMAKER_VERSION = "0.3.0"

# Set global vars
RDKIT_VERSION = os.environ.get("RDKIT_VERSION")
PYTHON_VERSION = os.environ.get("PYTHON_VERSION")
CXX11_ABI = os.environ.get("CUIKMOLMAKER_CXX11_ABI")
PUBLISH_TARGET = os.environ.get("PUBLISH_TARGET", "nvidia_pypi")
SYSTEM = platform.system()

# PUBLISH_TARGET controls the distribution name and version of the wheel:
#   - "nvidia_pypi" -> name="cuik_molmaker",     version=CUIK_MOLMAKER_VERSION
#   - "pypi"        -> name="cuik_molmaker_pin", version=RDKIT_VERSION
SUPPORTED_PUBLISH_TARGETS = ("nvidia_pypi", "pypi")
if PUBLISH_TARGET not in SUPPORTED_PUBLISH_TARGETS:
    print(
        f"Error: Unsupported PUBLISH_TARGET={PUBLISH_TARGET!r}. "
        f"Must be one of {SUPPORTED_PUBLISH_TARGETS}."
    )
    sys.exit(1)



def _build_prefix():
    """Return the prefix holding the RDKit that cuik will link against.

    CONDA_PREFIX, not sys.prefix: pip and uv build in an isolated virtual environment, so
    sys.prefix points at a throwaway venv that contains no RDKit. CMAKE_PREFIX_PATH below is
    set from the same variable, so this keeps detection and linkage in agreement.
    """
    return os.environ.get("CONDA_PREFIX") or sys.prefix


def _detect_pip_rdkit_libdir():
    """Return the PyPI rdkit wheel's bundled library directory, or None if RDKit is conda's.

    Linking the conda RDKit into a process whose Python-level rdkit is the PyPI wheel puts two
    RDKit builds in memory, and the conda ones are built against a newer libstdc++ than a
    manylinux wheel assumes. When the PyPI wheel is what is installed, link against its
    libraries instead: one RDKit in the process, and no C++ runtime newer than manylinux.
    Headers still come from the conda librdkit-dev package, since the wheel ships none.
    """
    import glob

    for site_packages in glob.glob(os.path.join(_build_prefix(), "lib", "python*", "site-packages")):
        libdir = os.path.join(site_packages, "rdkit.libs")
        if os.path.isdir(libdir) and glob.glob(os.path.join(libdir, "libRDKitGraphMol*")):
            return libdir
    return None


def _detect_rdkit_version():
    """Return the RDKit version cuik will link against, or None if it cannot be determined.

    Read from the shared library's soname rather than by importing rdkit, because pip
    builds in an isolated environment where rdkit is not importable but the libraries cuik
    compiles against are still present in the build prefix.
    """
    import glob
    import re

    for lib in glob.glob(os.path.join(_build_prefix(), "lib", "libRDKitRDGeneral.so.*")):
        match = re.search(r"\.so\.\d+\.(\d{4}\.\d+\.\d+)$", lib)
        if match:
            return match.group(1)
    try:
        import rdkit

        return rdkit.__version__
    except ImportError:
        return None


def _rdkit_uses_cxx11_abi():
    """Report whether the installed RDKit was built with the C++11 std::string ABI.

    Linking cuik with the other ABI produces a module that imports but cannot resolve
    RDKit's symbols, so this is detected rather than defaulted. libstdc++ tags the names of
    symbols whose type involves std::__cxx11::string with `B5cxx11`; any such export is
    conclusive, and looking for any of them rather than one named symbol keeps the check
    working across RDKit releases that move symbols between libraries.
    """
    import glob
    import subprocess

    for lib in glob.glob(os.path.join(_build_prefix(), "lib", "libRDKit*.so*")):
        try:
            symbols = subprocess.run(
                ["nm", "-D", "--defined-only", lib], capture_output=True, text=True, check=True
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue
        if "B5cxx11" in symbols:
            return True
    return False


class CMakeBuild(build_ext):
    def run(self):

        # Detect if we're doing an install against pip rdkit
        cmake_extra_args = []
        cuikmolmaker_build_against_pip = os.getenv(
            "CUIKMOLMAKER_BUILD_AGAINST_PIP_RDKIT"
        )
        # Fall back to detection so that a plain `pip install` picks the right RDKit without
        # the caller having to know which one is installed.
        if cuikmolmaker_build_against_pip is None:
            detected_libdir = _detect_pip_rdkit_libdir()
            if detected_libdir:
                print(f"Detected PyPI RDKit libraries at {detected_libdir}; linking against them")
                cuikmolmaker_build_against_pip = "1"
                os.environ.setdefault("CUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR", detected_libdir)
                os.environ.setdefault(
                    "CUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR",
                    os.path.join(_build_prefix(), "include", "rdkit"),
                )
                os.environ.setdefault(
                    "CUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR",
                    os.path.join(_build_prefix(), "include"),
                )
        cmake_extra_args.extend(
            [
                f"-DCUIKMOLMAKER_CXX11_ABI={CXX11_ABI}",
            ]
        )
        if cuikmolmaker_build_against_pip:
            CUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR = os.getenv(
                "CUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR"
            )
            if not CUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR:
                raise ValueError(
                    "CUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR must be set when "
                    "building against pip rdkit"
                )
            CUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR = os.getenv(
                "CUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR"
            )
            if not CUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR:
                raise ValueError(
                    "CUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR must be set when "
                    "building against pip rdkit"
                )
            CUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR = os.getenv(
                "CUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR"
            )
            if not CUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR:
                raise ValueError(
                    "CUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR must be set "
                    "when building against pip rdkit"
                )
            cmake_extra_args.extend(
                [
                    "-DCUIKMOLMAKER_BUILD_AGAINST_PIP_RDKIT=ON",
                    f"-DCUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR="
                    f"{CUIKMOLMAKER_BUILD_AGAINST_PIP_LIBDIR}",
                    f"-DCUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR="
                    f"{CUIKMOLMAKER_BUILD_AGAINST_PIP_INCDIR}",
                    f"-DCUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR="
                    f"{CUIKMOLMAKER_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR}",
                ]
            )
        # Prepare platform-specific CMake command
        platform_name = platform.system()

        if platform_name == "Windows":
            cmake_prefix_path = os.environ["CONDA_PREFIX"]
            cmake_extra_args.extend(
                ["-DCMAKE_BUILD_TYPE=Release", "-G", "Ninja", "-S", ".", "-B", "build"]
            )

            cmake_cmd = [
                "cmake",
                f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            ] + cmake_extra_args

            # Run CMake configure
            print("Running CMake configure command:", " ".join(cmake_cmd))
            subprocess.check_call(
                cmake_cmd, cwd=os.path.abspath(os.path.dirname(__file__))
            )

            # Run CMake build
            cmake_build_cmd = ["cmake", "--build", "build", "-j", "4"]
            print("Running CMake build command:", " ".join(cmake_build_cmd))
            subprocess.check_call(
                cmake_build_cmd, cwd=os.path.abspath(os.path.dirname(__file__))
            )

        elif platform_name in ("Linux", "Darwin"):

            # Ensure build directory exists
            build_dir = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "build"
            )
            os.makedirs(build_dir, exist_ok=True)

            # Prepare cmake command
            cmake_args = [
                "cmake",
                f"-DCMAKE_PREFIX_PATH={os.environ['CONDA_PREFIX']}",
            ]
            cmake_args.extend(cmake_extra_args)
            cmake_args.append(os.path.abspath(os.path.dirname(__file__)))

            # Run CMake configure
            print("Running CMake:", " ".join(cmake_args))
            subprocess.check_call(cmake_args, cwd=build_dir)

            # Run make
            print("Running make -j4")
            subprocess.check_call(["make", "-j4"], cwd=build_dir)
        else:
            raise ValueError(f"Unsupported platform: {platform_name}")

        # Call the original build_ext to copy .so files, etc.
        super().run()

        self._install_artifacts()

    def build_extension(self, ext):
        """Skip setuptools' own compilation; CMake has already produced the real modules."""

    def _install_artifacts(self):
        """Place the freshly built extension and core library where the wheel will find them.

        Done here rather than at module scope so that a one-shot `pip install` works: setup.py
        is executed for metadata before any build has run, so a module-level check can only
        ever see a stale artifact or none at all. The files are written both into the source
        package -- for editable and in-place builds -- and into build_lib, because build_py
        has already staged the package by the time the extension exists.
        """
        artifacts = (
            (os.path.join("build", f"cuik_molmaker_cpp.{so_suffix}"), dest_dir, "compiled extension"),
            (lib_file, lib_dir, "shared library"),
        )
        roots = [""]
        if getattr(self, "build_lib", None):
            roots.append(self.build_lib)

        for source, destination, what in artifacts:
            if not os.path.exists(source):
                raise FileNotFoundError(f"{what} missing after the CMake build: {source}")
            for root in roots:
                target = os.path.join(root, destination) if root else destination
                os.makedirs(target, exist_ok=True)
                print(f"Copying {what} to {target}")
                shutil.copy2(source, target)


# Release builds pin these explicitly so the wheel is tagged for one RDKit and Python
# combination. A plain `pip install` has no such wheel to tag, so fall back to whatever the
# build environment provides; that is also the only combination the result can be used with.
if RDKIT_VERSION is None:
    RDKIT_VERSION = _detect_rdkit_version()
if RDKIT_VERSION is None:
    print("Error: could not determine the RDKit version to build against.")
    print("Specify it explicitly:")
    print(
        "RDKIT_VERSION=2026.03.4 PYTHON_VERSION=3.13 CUIKMOLMAKER_CXX11_ABI=ON "
        "python setup.py build_ext --inplace"
    )
    sys.exit(1)

if PYTHON_VERSION is None:
    PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

if CXX11_ABI is None:
    # RDKit builds that expose std::string in their ABI are the ones cuik links against, so
    # match the ABI of the RDKit actually installed rather than guessing a default.
    CXX11_ABI = "ON" if _rdkit_uses_cxx11_abi() else "OFF"

# Update setup.cfg with the Python tag
PYTHON_DIGIT_ONLY_VERSION = PYTHON_VERSION.replace(".", "")

config = configparser.ConfigParser()
config.read("setup.cfg")
if "bdist_wheel" not in config:
    config["bdist_wheel"] = {}
config["bdist_wheel"]["python-tag"] = f"py{PYTHON_DIGIT_ONLY_VERSION}"
config["bdist_wheel"]["plat_name"] = sysconfig.get_platform()
with open("setup.cfg", "w") as f:
    config.write(f)

if PUBLISH_TARGET == "pypi":
    PACKAGE_NAME = "cuik_molmaker_pin"
    PACKAGE_VERSION = RDKIT_VERSION
elif PUBLISH_TARGET == "nvidia_pypi":
    PACKAGE_NAME = "cuik_molmaker"
    PACKAGE_VERSION = CUIK_MOLMAKER_VERSION
else:
    raise ValueError(f"Unsupported PUBLISH_TARGET: {PUBLISH_TARGET}")

print(
    f"Building with RDKIT_VERSION={RDKIT_VERSION}, "
    f"PYTHON_VERSION={PYTHON_VERSION}, "
    f"CXX11_ABI={CXX11_ABI}, "
    f"PUBLISH_TARGET={PUBLISH_TARGET}, "
    f"PACKAGE_NAME={PACKAGE_NAME}, "
    f"PACKAGE_VERSION={PACKAGE_VERSION}"
)

# Create package directory structure first
dest_dir = os.path.join("cuik_molmaker")
utils_dir = os.path.join(dest_dir, "utils")
data_dir = os.path.join(dest_dir, "data")

# Set appropriate file extensions based on system
if SYSTEM == "Darwin":  # macOS
    so_suffix = f"cpython-{PYTHON_DIGIT_ONLY_VERSION}-darwin.so"
    lib_extension = "dylib"
    lib_dir = os.path.join(dest_dir, "lib")
    lib_file = os.path.join("build", f"libcuik_molmaker_core.{lib_extension}")
elif SYSTEM == "Linux":
    so_suffix = f"cpython-{PYTHON_DIGIT_ONLY_VERSION}-x86_64-linux-gnu.so"
    lib_extension = "so"
    lib_dir = os.path.join(dest_dir, "lib")
    lib_file = os.path.join("build", f"libcuik_molmaker_core.{lib_extension}")
elif SYSTEM == "Windows":
    machine = platform.machine().lower()  # like AMD64
    so_suffix = f"cp{PYTHON_DIGIT_ONLY_VERSION}-win_{machine}.pyd"
    lib_extension = "dll"
    lib_file = os.path.join("build", f"cuik_molmaker_core.{lib_extension}")
    # On Windows, DLLs should be in the same directory or on PATH
    lib_dir = dest_dir
else:
    raise ValueError(f"Unsupported platform: {SYSTEM}")

os.makedirs(dest_dir, exist_ok=True)
os.makedirs(lib_dir, exist_ok=True)
os.makedirs(utils_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Copy all Python files from src/ to package directory
src_py_files = glob.glob(os.path.join("src", "**", "*.py"), recursive=True)
for src_file in src_py_files:
    # Get relative path from src/
    rel_path = os.path.relpath(src_file, "src")
    # Create destination path
    dest_path = os.path.join(dest_dir, rel_path)
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Copy the file
    print(f"Copying {src_file} to {dest_path}")
    shutil.copy2(src_file, dest_path)

# Copy data files
data_files = [
    "best_normalization_params.json",
    "fast_normalization_params.json",
    "descriptastorus_normalization_params.json",
    "README.md",
]
for data_file in data_files:
    src_path = os.path.join("data", data_file)
    dest_path = os.path.join(data_dir, data_file)
    if os.path.exists(src_path):
        print(f"Copying {src_path} to {dest_path}")
        shutil.copy2(src_path, dest_path)
    else:
        print(f"WARNING: {src_path} not found")


# Create an empty __init__.py in the lib directory to make it a package
lib_init_file = os.path.join(lib_dir, "__init__.py")
if not os.path.exists(lib_init_file):
    print(f"Creating {lib_init_file}")
    with open(lib_init_file, "w") as f:
        f.write("# This file makes the lib directory a Python package\n")


setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    author="S. Veccham",
    author_email="sveccham@nvidia.com",
    description="C++ module for featurizing molecules",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    # Include both src directory and cuik_molmaker_py package
    packages=find_packages(
        include=[
            "cuik_molmaker",
            "cuik_molmaker.lib",
            "cuik_molmaker.utils",
            "cuik_molmaker.data",
        ]
    ),
    package_data={
        "cuik_molmaker": [
            "*.so",
            "*.pyd",
            "*.dll",
            "*.py",
            "data/*.json",
            "data/*.md",
        ],  # Include Python extension and Python files
        "cuik_molmaker.lib": [
            "*.so",
            "__init__.py",
            "*.dll",
            "*.dylib",
        ],  # Include shared libraries and __init__.py
        "cuik_molmaker.utils": ["*.py"],  # Include Python files
    },
    include_package_data=True,
    # CMakeBuild ignores this and drives CMake instead, but setuptools only runs build_ext
    # when the distribution declares an extension, and without that a `pip install` produces
    # a wheel with no compiled artifacts in it.
    ext_modules=[Extension("cuik_molmaker._cmake_placeholder", sources=[])],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    install_requires=[
        f"rdkit=={RDKIT_VERSION}",
        "scipy",
        "pandas",
    ],
    build_requires=[
        f"rdkit=={RDKIT_VERSION}",
    ],
    tests_require=["pytest"],
    extras_require={
        "dev": [
            "black>=24.2.0",
            "flake8>=7.3.0",
            "isort>=5.13.2",
            "pre-commit>=3.6.0",
            "bump2version>=1.0.1",
        ],
    },
    python_requires=">=3.11,<3.15",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    entry_points={
        "console_scripts": [
            "cuik-molmaker-fit-distribution=cuik_molmaker.utils.fit_distribution:main",
            "cuik-molmaker-mol-features=cuik_molmaker.mol_features:main",
        ],
    },
)

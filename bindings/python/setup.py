#!/usr/bin/env python3

import os
import platform
import re
import subprocess
import sys

from packaging import version
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# Environment variables:
# - `USE_CUDA=0` disables building CUDA components
# - `USE_KENLM=0` disables building KenLM
# - `USE_MKL=1` enables MKL (may cause errors)


def check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def check_negative_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["OFF", "0", "NO", "FALSE", "N"]


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = re.search(r"version\s*([\d.]+)", out.decode().lower()).group(1)
        if version.parse(cmake_version) < version.parse("3.5.1"):
            raise RuntimeError("CMake >= 3.5.1 is required to build wav2letter")

        # our CMakeLists builds all the extensions at once
        self.build_extensions()

    def build_extensions(self):
        extdir = os.path.abspath("wav2letter")
        sourcedir = os.path.abspath("../..")
        use_cuda = "OFF" if check_negative_env_flag("USE_CUDA") else "ON"
        use_kenlm = "OFF" if check_negative_env_flag("USE_KENLM") else "ON"
        use_mkl = "ON" if check_env_flag("USE_MKL") else "OFF"
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DW2L_BUILD_LIBRARIES_ONLY=ON",
            "-DW2L_BUILD_FOR_PYTHON=ON",
            "-DW2L_LIBRARIES_USE_CUDA=" + use_cuda,
            "-DW2L_LIBRARIES_USE_KENLM=" + use_kenlm,
            "-DW2L_LIBRARIES_USE_MKL=" + use_mkl,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            # cmake_args += [
            #     "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            # ]
            # if sys.maxsize > 2 ** 32:
            #     cmake_args += ["-A", "x64"]
            # build_args += ["--", "/m"]
            raise RuntimeError("wav2letter doesn't support building on Windows yet")
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            subprocess.check_call(
                ["cmake", sourcedir] + cmake_args, cwd=self.build_temp, env=env
            )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="wav2letter",
    version="0.0.2",
    author="Jeff Cai",
    author_email="jcai@fb.com, antares@fb.com",
    description="wav2letter bindings for python",
    long_description="",
    ext_modules=[
        CMakeExtension("wav2letter._common"),
        CMakeExtension("wav2letter._criterion"),
        CMakeExtension("wav2letter._decoder"),
        CMakeExtension("wav2letter._feature"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

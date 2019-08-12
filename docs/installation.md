# Building wav2letter++

## Dependencies

### 1. flashlight

`wav2letter++` uses **[flashlight](https://github.com/facebookresearch/flashlight/)** as its core ML backend.
- Please follow the provided [install procedures](https://fl.readthedocs.io/en/latest/installation.html).
- wav2letter++ requires flashlight built with distributed training enabled (default).

### 2. KenLM

`wav2letter++` uses [KenLM](https://github.com/kpu/kenlm) to allow beam-search decoding with an n-gram language model.
- _At least one_ of LZMA, BZip2, or Z is required for LM compression with KenLM.
- It is highly recommended to build KenLM with position-independent code (`-fPIC`) enabled, to enable python compatibility.
- After installing, run `export KENLM_ROOT_DIR=...` so that `wav2letter++` can find it. This is needed because KenLM doesn't support a `make install` step.

Example build commands on Ubuntu:

    sudo apt-get install liblzma-dev libbz2-dev libzstd-dev
    git clone https://github.com/kpu/kenlm.git
    cd kenlm
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=20 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    make -j16
    # don't forget to export KENLM_ROOT_DIR

### 3. Additional Dependencies

The following additional packages are required:
- Any CBLAS library, i.e. _at least one_ of these:
	- [ATLAS](http://math-atlas.sourceforge.net/)
	- [OpenBLAS](https://www.openblas.net/)
	- [Accelerate](https://developer.apple.com/documentation/accelerate/blas)
	- [Intel MKL](https://software.intel.com/en-us/mkl) (used preferentially if present)
- [FFTW3](http://www.fftw.org/)
- [libsndfile](https://github.com/erikd/libsndfile)
	- Should be built with `Ogg`, `Vorbis`, and `FLAC` libraries.
- [gflags](https://github.com/gflags/gflags)
- [glog](https://github.com/google/glog)

Example (Ubuntu). The following command will install all the above packages:

    apt-get install libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev

### 4. Optional Notes

The following dependencies should be already installed for flashlight:
- A C++ compiler with good C++11 support (e.g. g++ >= 4.8)
- [cmake](https://cmake.org/) >= 3.5.1, and `make`
- [CUDA](https://developer.nvidia.com/cuda-downloads) >= 9.2, only if using CUDA backend

The following dependencies are automatically downloaded and built by cmake:
- [gtest and gmock](https://github.com/google/googletest) 1.8.1, only if building tests
- [CUB](https://github.com/NVlabs/cub) 1.8.0, only if using CUDA backend

The following dependencies are optional:
- [OpenMP](https://www.openmp.org/), if present, will be used for better performance.

## Build Options
| Option                   | Configuration       | Default Value |
|--------------------------|---------------------|---------------|
| W2L_BUILD_LIBRARIES_ONLY | ON, OFF             | OFF           |
| W2L_LIBRARIES_USE_CUDA   | ON, OFF             | ON            |
| W2L_LIBRARIES_USE_KENLM  | ON, OFF             | ON            |
| W2L_LIBRARIES_USE_MKL    | ON, OFF             | ON            |
| W2L_BUILD_FOR_PYTHON     | ON, OFF             | OFF           |
| W2L_BUILD_TESTS          | ON, OFF             | ON            |
| W2L_BUILD_EXAMPLES       | ON, OFF             | ON            |
| W2L_BUILD_EXPERIMENTAL   | ON, OFF             | OFF           |
| W2L_BUILD_SCRIPTS        | ON, OFF             | OFF           |
| CMAKE_BUILD_TYPE         | <CMake build types> | Debug         |

## General Build Instructions
First, clone the repository:
```
git clone --recursive https://github.com/facebookresearch/wav2letter.git
```
and follow the build instructions for your specific OS.

There is no `install` procedure currently supported for wav2letter++. Building
produces three binaries in the `build` directory:
- `Train`: given a dataset of input audio and corresponding transcriptions in
  sub-word units (graphemes, phonemes, etc), trains the acoustic model.
- `Test`: performs inference on a given dataset with an acoustic model.
- `Decode`: given an acoustic model/pre-computed network emissions and a
  language model, computes the most likely sequence of words for a given
  dataset.

### Building on Linux
wav2letter++ has been tested on many Linux distributions including Ubuntu, Debian, CentOS, Amazon Linux, and RHEL.

Assuming you have [ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux), [flashlight](https://fl.readthedocs.io/en/latest/installation.html), [libsndfile](https://github.com/erikd/libsndfile#hacking), and [KenLM](https://github.com/kpu/kenlm#compiling) built/installed, install the below dependencies with `apt` (or your distribution's package manager):
```
sudo apt-get update
sudo apt-get install \
    # Audio encoding libs for libsndfile \
    libasound2-dev \
    libflac-dev \
    libogg-dev \
    libtool \
    libvorbis-dev \
    # FFTW for Fourier transforms \
    libfftw3-dev \
    # Compression libraries for KenLM \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libboost-all-dev \
    # gflags \
    libgflags-dev \
    libgflags2v5 \
    # glog \
    libgoogle-glog-dev \
    libgoogle-glog0v5 \
```

MKL and KenLM aren't easily discovered by CMake by default; export environment variables to make sure they're found. On most Linux-based systems, MKL is installed in `/opt/intel/mkl`. Since KenLM doesn't support an install step, [after building KenLM](https://github.com/kpu/kenlm#compiling), point CMake to wherever you downloaded and built KenLM:
```
export MKLROOT=/opt/intel/mkl # or path to MKL
export KENLM_ROOT_DIR=[path to KenLM]
```

Once you've downloaded wav2letter++ and built and installed the required dependencies:
```
# in your wav2letter++ directory
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4 # (or any number of threads)
```

### Building/Running with Docker

wav2letter++ and its dependencies can also be built with the provided Dockerfile. Both CUDA and CPU backends are supported with Docker

To build wav2letter++ with Docker:
- Install [Docker](https://docs.docker.com/engine/installation/) and, if using the CUDA backend, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/)
- Run the docker image with CUDA/CPU backend in a new container:

  ```
  # with CUDA backend
  sudo docker run --runtime=nvidia --rm -itd --ipc=host --name w2l wav2letter/wav2letter:cuda-latest
  # or with CPU backend
  sudo docker run --rm -itd --ipc=host --name w2l wav2letter/wav2letter:cpu-latest
  sudo docker exec -it w2l bash
  ```

- To run tests inside a container

  ```
  cd /root/wav2letter/build && make test
  ```

- Build Docker image from the source (using `--no-cache` will provide the latest version of `flashlight` inside the image if you have built the image previously for earlier versions of `wav2letter`):

  ```
  git clone --recursive https://github.com/facebookresearch/wav2letter.git
  cd wav2letter
  # for CUDA backend
  sudo docker build --no-cache -f ./Dockerfile-CUDA -t wav2letter .
  # for CPU backend
  sudo docker build --no-cache -f ./Dockerfile-CPU -t wav2letter .
  ```

  For logging during training/testing/decoding inside a container, use the `--logtostderr=1` flag.

## Building Python bindings

### Dependencies

We require `python` >= 3.6 with the following packages installed:
- [packaging](https://pypi.org/project/packaging/)
- [torch](https://pypi.org/project/torch/)

[Anaconda](https://www.anaconda.com/distribution/) makes this easy. There are plenty of tutorials on how to set this up.

Aside from the above, the dependencies for Python bindings are a **strict subset** of the dependencies for the main wav2letter++ build. So if you already have the dependencies to build wav2letter++, you're all set to build python bindings as well.

The following dependencies **are required** to build python bindings:
- [KenLM](https://github.com/kpu/kenlm)
- [ATLAS](http://math-atlas.sourceforge.net/) or [OpenBLAS](https://www.openblas.net/)
- [FFTW3](http://www.fftw.org/)
- [cmake](https://cmake.org/) >= 3.5.1, and `make`
- [CUDA](https://developer.nvidia.com/cuda-downloads) >= 9.2

Please refer to the previous sections for details on how to install the above dependencies.

The following dependencies **are not required** to build python bindings:
- flashlight
- libsndfile
- gflags
- glog

### Build Instructions

Once the dependencies are satisfied, simply run from wav2letter root:

    cd bindings/python
    pip install -e .

Note that if you encounter errors, you'll probably have to `rm -rf build` before retrying the install.

### Advanced Options

The following environment variables can be used to control various options:
- `USE_CUDA=0` removes the CUDA dependency, but you won't be able to use ASG criterion with CUDA tensors.
- `USE_KENLM=0` removes the KenLM dependency, but you won't be able to use the decoder unless you write C++ pybind11 bindings for your own LM.
- `USE_MKL=1` will use Intel MKL for featurization but this may cause dynamic loading conflicts.
- If you do not have `torch`, you'll only have a raw pointer interface to ASG criterion instead of `class ASGLoss(torch.nn.Module)`.

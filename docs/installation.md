# Building wav2letter++

## Build Requirements
- A C++ compiler with good C++ 11 support (e.g. g++ >= 4.8)
- [cmake](https://cmake.org/) — version 3.5.1 or later, make

## Dependencies
- [flashlight](https://github.com/facebookresearch/flashlight/) is required.
  flashlight _must_ be built with distributed training enabled.
- [libsndfile](https://github.com/erikd/libsndfile) is required for loading
  audio. If using wav2letter++ with `flac` files, `libsndfile`
  [must be built](https://github.com/erikd/libsndfile#requirements) with `Ogg`,
  `Vorbis` and `FLAC` libraries.
- Any CBLAS library, such as
  [ATLAS](http://math-atlas.sourceforge.net/),
  [OpenBLAS](https://www.openblas.net/),
  [Accelerate](https://developer.apple.com/documentation/accelerate/blas), or
  [Intel MKL](https://software.intel.com/en-us/mkl),
  is required for featurization. Intel MKL will be used preferentially unless
  otherwise specified.
- [FFTW](http://www.fftw.org/) is required for featurization.
- [KenLM](https://github.com/kpu/kenlm) is required for the decoder. One of
  LZMA, BZip2, or Z is required for LM compression with KenLM.
  * **NB:** KenLM should be built with position-independent code (`-fPIC`) enabled, otherwise wav2letter++ python bindings for decoder will not work.
  * In KenLM build directory: `cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=6 -DCMAKE_POSITION_INDEPENDENT_CODE=ON`
- [gflags](https://github.com/gflags/gflags) is required.
- [glog](https://github.com/google/glog) is required.

The following dependencies are automatically downloaded/built on build:
- [gtest and gmock](https://github.com/google/googletest) 1.8.1 is built if
  building tests.
- If using the CUDA criterion backend (see below), [NVIDIA cub](https://github.com/NVlabs/cub) 1.8.0 is downloaded and linked to criterion CUDA kernels.

### Optional Dependencies
- If `flashlight` was built with CUDA backend, then CUDA >= 9.2 is required to build custom CUDA kernels for wav2letter++ criterions. Using [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive) is recommended.
- wav2letter++ will try to compile with [OpenMP](https://www.openmp.org/) for better performance.

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
wav2letter++ has been tested on Ubuntu 16.04 and CentOS 7.5.

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

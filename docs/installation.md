# Building wav2letter++

## Build Requirements
- A C++ compiler with good C++ 11 support (e.g. g++ >= 4.8)
- [cmake](https://cmake.org/) — version 3.5.1 or later, make

## Dependencies
- [flashlight](https://github.com/facebookresearch/flashlight/) is required. flashlight _must_ be built with distributed training enabled.
- [libsndfile](https://github.com/erikd/libsndfile) is required for loading audio. If using wav2letter++ with `flac` files, `libsndfile` [must be built](https://github.com/erikd/libsndfile#requirements) with `Ogg`, `Vorbis` and `FLAC` libraries.
- [Intel's Math Kernel Library](https://software.intel.com/en-us/mkl) is required for featurization.
- [FFTW](http://www.fftw.org/) is required for featurization.
- [KenLM](https://github.com/kpu/kenlm) is required for the decoder. One of LZMA, BZip2, or Z is required for LM compression with KenLM.
- [glags](https://github.com/gflags/gflags) is required.
- [glog](https://github.com/google/glog) is required.
- [Google Test](https://github.com/google/googletest) >= 1.8.0 is required if building tests.

### Optional Dependencies
- `flashlight` requires CUDA >= 9.2; if building wav2letter++ with the `CUDA` criterion backend, CUDA >= 9.2 is required. Using [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive) is recommended.
- If building with the `CPU` criteiron backend, wav2letter++ will try to compile with [OpenMP](https://www.openmp.org/), for better performance.

## Build Options
| Options           | Configuration     | Default Value |
|-------------------|-------------------|---------------|
| CRITERION_BACKEND | CUDA, CPU         | CUDA          |
| BUILD_TESTS       | ON, OFF           | ON            |
| CMAKE_BUILD_TYPE  | CMake build types | Debug         |

## General Build Instructions
First, clone the repository:
```
git clone --recursive https://github.com/facebookresearch/wav2letter.git
```
and follow the build instructions for your specific OS.

There is no `install` procedure currently supported for wav2letter++. Building produces three binaries in the `build` directory:
- `Train`: given a dataset of input audio and corresponding transcriptions in sub-word units (graphemes, phonemes, etc), trains the acoustic model.
- `Test`: performs inference on a given dataset with an acoustic model.
- `Decode`: given an acoustic model/pre-computed network emissions and a language model, computes the most likely sequence of words for a given dataset.

### Building on Linux
wav2letter++ has been tested on Ubuntu 16.04 and CentOS 7.5.

Building on Linux is simple. Once you've downloaded wav2letter++ and built and installed the required dependencies:
```
# in your wav2letter++ directory
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCRITERION_BACKEND=[backend]
make -j4 # (or any number of threads)
```

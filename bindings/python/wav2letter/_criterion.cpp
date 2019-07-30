/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>

#include "libraries/criterion/cpu/ForceAlignmentCriterion.h"
#include "libraries/criterion/cpu/FullConnectionCriterion.h"
#include "libraries/criterion/cpu/ViterbiPath.h"

#ifdef W2L_LIBRARIES_USE_CUDA
#include "libraries/criterion/cuda/ForceAlignmentCriterion.cuh"
#include "libraries/criterion/cuda/FullConnectionCriterion.cuh"
#include "libraries/criterion/cuda/ViterbiPath.cuh"
#endif // W2L_LIBRARIES_USE_CUDA

namespace py = pybind11;
using namespace w2l;

template <class T>
static T castBytes(const py::bytes& b) {
  static_assert(
      std::is_standard_layout<T>::value,
      "types represented as bytes must be standard layout");
  std::string s = b;
  if (s.size() != sizeof(T)) {
    throw std::runtime_error("wrong py::bytes size to represent object");
  }
  return *reinterpret_cast<const T*>(s.data());
}

using CpuFAC = cpu::ForceAlignmentCriterion<float>;
using CpuFCC = cpu::FullConnectionCriterion<float>;
using CpuViterbi = cpu::ViterbiPath<float>;

static void CpuFAC_forward(
    int B,
    int T,
    int N,
    int L,
    CriterionScaleMode scaleMode,
    py::bytes input,
    py::bytes target,
    py::bytes targetSize,
    py::bytes trans,
    py::bytes loss,
    py::bytes workspace) {
  CpuFAC::forward(
      B,
      T,
      N,
      L,
      scaleMode,
      castBytes<const float*>(input),
      castBytes<const int*>(target),
      castBytes<const int*>(targetSize),
      castBytes<const float*>(trans),
      castBytes<float*>(loss),
      castBytes<void*>(workspace));
}

static void CpuFAC_backward(
    int B,
    int T,
    int N,
    int L,
    py::bytes target,
    py::bytes targetSize,
    py::bytes grad,
    py::bytes inputGrad,
    py::bytes transGrad,
    py::bytes workspace) {
  CpuFAC::backward(
      B,
      T,
      N,
      L,
      castBytes<const int*>(target),
      castBytes<const int*>(targetSize),
      castBytes<const float*>(grad),
      castBytes<float*>(inputGrad),
      castBytes<float*>(transGrad),
      castBytes<void*>(workspace));
}

static void CpuFCC_forward(
    int B,
    int T,
    int N,
    CriterionScaleMode scaleMode,
    py::bytes input,
    py::bytes targetSize,
    py::bytes trans,
    py::bytes loss,
    py::bytes workspace) {
  CpuFCC::forward(
      B,
      T,
      N,
      scaleMode,
      castBytes<const float*>(input),
      castBytes<const int*>(targetSize),
      castBytes<const float*>(trans),
      castBytes<float*>(loss),
      castBytes<void*>(workspace));
}

static void CpuFCC_backward(
    int B,
    int T,
    int N,
    py::bytes trans,
    py::bytes grad,
    py::bytes inputGrad,
    py::bytes transGrad,
    py::bytes workspace) {
  CpuFCC::backward(
      B,
      T,
      N,
      castBytes<const float*>(trans),
      castBytes<const float*>(grad),
      castBytes<float*>(inputGrad),
      castBytes<float*>(transGrad),
      castBytes<void*>(workspace));
}

static void CpuViterbi_compute(
    int B,
    int T,
    int N,
    py::bytes input,
    py::bytes trans,
    py::bytes path,
    py::bytes workspace) {
  CpuViterbi::compute(
      B,
      T,
      N,
      castBytes<const float*>(input),
      castBytes<const float*>(trans),
      castBytes<int*>(path),
      castBytes<void*>(workspace));
}

#ifdef W2L_LIBRARIES_USE_CUDA

using CudaFAC = cuda::ForceAlignmentCriterion<float>;
using CudaFCC = cuda::FullConnectionCriterion<float>;
using CudaViterbi = cuda::ViterbiPath<float>;

static void CudaFAC_forward(
    int B,
    int T,
    int N,
    int L,
    CriterionScaleMode scaleMode,
    py::bytes input,
    py::bytes target,
    py::bytes targetSize,
    py::bytes trans,
    py::bytes loss,
    py::bytes workspace,
    py::bytes stream) {
  CudaFAC::forward(
      B,
      T,
      N,
      L,
      scaleMode,
      castBytes<const float*>(input),
      castBytes<const int*>(target),
      castBytes<const int*>(targetSize),
      castBytes<const float*>(trans),
      castBytes<float*>(loss),
      castBytes<void*>(workspace),
      castBytes<cudaStream_t>(stream));
}

static void CudaFAC_backward(
    int B,
    int T,
    int N,
    int L,
    py::bytes target,
    py::bytes targetSize,
    py::bytes grad,
    py::bytes inputGrad,
    py::bytes transGrad,
    py::bytes workspace,
    py::bytes stream) {
  CudaFAC::backward(
      B,
      T,
      N,
      L,
      castBytes<const int*>(target),
      castBytes<const int*>(targetSize),
      castBytes<const float*>(grad),
      castBytes<float*>(inputGrad),
      castBytes<float*>(transGrad),
      castBytes<void*>(workspace),
      castBytes<cudaStream_t>(stream));
}

static void CudaFCC_forward(
    int B,
    int T,
    int N,
    CriterionScaleMode scaleMode,
    py::bytes input,
    py::bytes targetSize,
    py::bytes trans,
    py::bytes loss,
    py::bytes workspace,
    py::bytes stream) {
  CudaFCC::forward(
      B,
      T,
      N,
      scaleMode,
      castBytes<const float*>(input),
      castBytes<const int*>(targetSize),
      castBytes<const float*>(trans),
      castBytes<float*>(loss),
      castBytes<void*>(workspace),
      castBytes<cudaStream_t>(stream));
}

static void CudaFCC_backward(
    int B,
    int T,
    int N,
    py::bytes trans,
    py::bytes grad,
    py::bytes inputGrad,
    py::bytes transGrad,
    py::bytes workspace,
    py::bytes stream) {
  CudaFCC::backward(
      B,
      T,
      N,
      castBytes<const float*>(trans),
      castBytes<const float*>(grad),
      castBytes<float*>(inputGrad),
      castBytes<float*>(transGrad),
      castBytes<void*>(workspace),
      castBytes<cudaStream_t>(stream));
}

static void CudaViterbi_compute(
    int B,
    int T,
    int N,
    py::bytes input,
    py::bytes trans,
    py::bytes path,
    py::bytes workspace,
    py::bytes stream) {
  CudaViterbi::compute(
      B,
      T,
      N,
      castBytes<const float*>(input),
      castBytes<const float*>(trans),
      castBytes<int*>(path),
      castBytes<void*>(workspace),
      castBytes<cudaStream_t>(stream));
}

#endif // W2L_LIBRARIES_USE_CUDA

PYBIND11_MODULE(_criterion, m) {
  py::enum_<CriterionScaleMode>(m, "CriterionScaleMode")
      .value("NONE", CriterionScaleMode::NONE)
      .value("INPUT_SZ", CriterionScaleMode::INPUT_SZ)
      .value("INPUT_SZ_SQRT", CriterionScaleMode::INPUT_SZ_SQRT)
      .value("TARGET_SZ", CriterionScaleMode::TARGET_SZ)
      .value("TARGET_SZ_SQRT", CriterionScaleMode::TARGET_SZ_SQRT);

  py::class_<CpuFAC>(m, "CpuForceAlignmentCriterion")
      .def("get_workspace_size", &CpuFAC::getWorkspaceSize)
      .def("forward", &CpuFAC_forward)
      .def("backward", &CpuFAC_backward);

  py::class_<CpuFCC>(m, "CpuFullConnectionCriterion")
      .def("get_workspace_size", &CpuFCC::getWorkspaceSize)
      .def("forward", &CpuFCC_forward)
      .def("backward", &CpuFCC_backward);

  py::class_<CpuViterbi>(m, "CpuViterbiPath")
      .def("get_workspace_size", &CpuViterbi::getWorkspaceSize)
      .def("compute", &CpuViterbi_compute);

#ifdef W2L_LIBRARIES_USE_CUDA
  m.attr("sizeof_cuda_stream") = py::int_(sizeof(cudaStream_t));

  py::class_<CudaFAC>(m, "CudaForceAlignmentCriterion")
      .def("get_workspace_size", &CudaFAC::getWorkspaceSize)
      .def("forward", &CudaFAC_forward)
      .def("backward", &CudaFAC_backward);

  py::class_<CudaFCC>(m, "CudaFullConnectionCriterion")
      .def("get_workspace_size", &CudaFCC::getWorkspaceSize)
      .def("forward", &CudaFCC_forward)
      .def("backward", &CudaFCC_backward);

  py::class_<CudaViterbi>(m, "CudaViterbiPath")
      .def("get_workspace_size", &CudaViterbi::getWorkspaceSize)
      .def("compute", &CudaViterbi_compute);
#endif // W2L_LIBRARIES_USE_CUDA
}

#pragma once

#include <cuda_runtime.h>

namespace w2l {
namespace cuda {

int fullConnectionCriterionForward(
    int T,
    int B,
    int N,
    const float* input,
    const float* trans,
    double* fccacc,
    cudaStream_t stream);

int fullConnectionCriterionBackward(
    int T,
    int B,
    int N,
    const float* trans,
    const double* fccacc,
    double* fccgacc,
    double* gtrans,
    cudaStream_t stream);

} // namespace cuda
} // namespace w2l

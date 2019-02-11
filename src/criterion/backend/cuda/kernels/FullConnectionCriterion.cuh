#pragma once

#include <cuda_runtime.h>

namespace w2l {
namespace cuda {

cudaError_t fullConnectionCriterionForward(
    int T,
    int B,
    int N,
    const float* input,
    const float* trans,
    double* transtmp,
    double* fccacc,
    cudaStream_t stream);

cudaError_t fullConnectionCriterionBackward(
    int T,
    int B,
    int N,
    const float* trans,
    double* transtmp,
    const double* fccacc,
    double* fccgacc,
    double* gtrans,
    cudaStream_t stream);

} // namespace cuda
} // namespace w2l

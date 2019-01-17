#pragma once

#include <cuda_runtime.h>

namespace w2l {
namespace cuda {

int viterbiPath(
    int T,
    int B,
    int N,
    const float* trans,
    float* alpha,
    int* beta,
    cudaStream_t stream);

} // namespace cuda
} // namespace w2l

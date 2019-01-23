#include "criterion/backend/cuda/kernels/ViterbiPath.cuh"

#include <cassert>
#include <cfloat>

#include <cub/cub.cuh>

namespace {

constexpr int kBlockSize = 128;

__global__ void viterbiStep(
    int N,
    const float* trans,
    const float* alpha_tp,
    float* alpha_t,
    int* beta_t) {
  using BlockReduce =
      cub::BlockReduce<cub::KeyValuePair<int, float>, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  assert(blockDim.x == kBlockSize);

  cub::KeyValuePair<int, float> threadMax(-1, -FLT_MAX);
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int k = blockIdx.x * N + i;
    float val = trans[k % (N * N)] + alpha_tp[k / (N * N) * N + i];
    if (val > threadMax.value) {
      threadMax.key = i;
      threadMax.value = val;
    }
  }

  auto result = BlockReduce(tempStorage).Reduce(threadMax, cub::ArgMax());
  if (threadIdx.x == 0) {
    alpha_t[blockIdx.x] += result.value;
    beta_t[blockIdx.x] = result.key;
  }
}

} // namespace

namespace w2l {
namespace cuda {

/**
 * equivalent arrayfire implementation (for batchsize 1):
   // pre: alpha = input
   array maxvals;
   array maxidxs;

   for (int t = 1; t < T; ++t) {
     max(maxvals, maxidxs, trans + tile(_alpha(span, t - 1), 1, N), 0);
     _alpha(span, t) += moddims(maxvals, N);
     _beta(span, t) = moddims(maxidxs, N);
   }
 */

int viterbiPath(
    int T,
    int B,
    int N,
    const float* trans,
    float* alpha,
    int* beta,
    cudaStream_t stream) {
  for (int t = 1; t < T; ++t) {
    viterbiStep<<<B * N, kBlockSize, 0, stream>>>(
        N, trans, alpha + (t - 1) * B * N, alpha + t * B * N, beta + t * B * N);
  }
  return 0;
}

} // namespace cuda
} // namespace w2l

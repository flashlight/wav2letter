#include "criterion/backend/cuda/kernels/FullConnectionCriterion.cuh"

#include <cassert>
#include <cfloat>
#include <cmath>

#include <cub/cub.cuh>

namespace {

constexpr int kBlockSize = 128;

__global__ void forwardStep(
    int N,
    const double* fccacc_tp,
    double* fccacc_t,
    const float* input_t,
    const float* trans,
    double* transtmp) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double maxValue;

  assert(blockDim.x == kBlockSize);

  double threadMax = -DBL_MAX;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int k = blockIdx.x * N + i;
    transtmp[k] = trans[k % (N * N)] + fccacc_tp[k / (N * N) * N + i];
    threadMax = (transtmp[k] > threadMax) ? transtmp[k] : threadMax;
  }

  double maxResult = BlockReduce(tempStorage).Reduce(threadMax, cub::Max());
  if (threadIdx.x == 0) {
    maxValue = maxResult;
  }
  __syncthreads();

  double threadSum = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int k = blockIdx.x * N + i;
    threadSum += exp(transtmp[k] - maxValue);
  }

  double sumResult = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) {
    fccacc_t[blockIdx.x] = log(sumResult) + maxValue + input_t[blockIdx.x];
  }
}

__global__ void backwardStep1(
    int N,
    const double* fccacc_tp,
    const double* fccgacc_t,
    const float* trans,
    double* transtmp,
    double* gtrans) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double maxValue;
  __shared__ double sumValue;

  assert(blockDim.x == kBlockSize);

  double threadMax = -DBL_MAX;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int k = blockIdx.x * N + i;
    transtmp[k] = trans[k % (N * N)] + fccacc_tp[k / (N * N) * N + i];
    threadMax = (transtmp[k] > threadMax) ? transtmp[k] : threadMax;
  }

  double maxResult = BlockReduce(tempStorage).Reduce(threadMax, cub::Max());
  if (threadIdx.x == 0) {
    maxValue = maxResult;
  }
  __syncthreads();

  double threadSum = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int k = blockIdx.x * N + i;
    transtmp[k] = exp(transtmp[k] - maxValue);
    threadSum += transtmp[k];
  }

  double sumResult = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) {
    sumValue = sumResult;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int k = blockIdx.x * N + i;
    transtmp[k] = transtmp[k] / sumValue * fccgacc_t[blockIdx.x];
    gtrans[k] += transtmp[k];
  }
}

// sums along dim 1 instead of 0
__global__ void
computeSums1(int N0, int N1, const double* transtmp, double* sums) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  assert(blockDim.x == kBlockSize);

  double threadSum = 0;
  for (int i = threadIdx.x; i < N1; i += blockDim.x) {
    threadSum +=
        transtmp[(blockIdx.x / N0) * N0 * N1 + i * N0 + (blockIdx.x % N0)];
  }

  double result = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) {
    sums[blockIdx.x] = result;
  }
}

} // namespace

namespace w2l {
namespace cuda {

/**
 * original arrayfire code for FCC forward:
 *
 * for (int t = 1; t < T; t++) {
 *   const auto& fccacc_tp = fccacc(span, span, t - 1); // [N, B, 1]
 *   const auto& transtmp = tile(trans, 1, 1, B) +
 *       tile(moddims(fccacc_tp, N, 1, B), 1, N); // [N, N, B]
 *   const auto& maxes = max(transtmp, 0); // [1, N, B]
 *   const auto& lse =
 *       maxes + log(sum(exp(transtmp - tile(maxes, N)), 0)); // [1, N, B]
 *   fccacc(span, span, t) = moddims(lse, N, B, 1) + inp(span, span, t);
 * }
 */

int fullConnectionCriterionForward(
    int T,
    int B,
    int N,
    const float* input,
    const float* trans,
    double* fccacc,
    cudaStream_t stream) {
  int retval = 0;
  double* transtmp;

  if ((retval = cudaMalloc(&transtmp, B * N * N * sizeof(double)))) {
    goto err_1;
  }

  for (int t = 1; t < T; ++t) {
    forwardStep<<<B * N, kBlockSize, 0, stream>>>(
        N,
        fccacc + (t - 1) * B * N,
        fccacc + t * B * N,
        input + t * B * N,
        trans,
        transtmp);
  }

  cudaFree(transtmp);
err_1:
  return retval;
}

/**
 * original arrayfire code for FCC backward:
 *
 * for (int t = T - 1; t > 0; t--) {
 *   const auto& fccacc_tp = fccacc(span, span, t - 1); // [N, B]
 *   const auto& transtmp = tile(trans, 1, 1, B) +
 *       tile(moddims(fccacc_tp, N, 1, B), 1, N); // [N, N, B]
 *   const auto& maxes = max(transtmp, 0); // [1, N, B]
 *   const auto& exps = exp(transtmp - tile(maxes, N)); // [N, N, B]
 *   const auto& dlse = exps / tile(sum(exps, 0), N); // [N, N, B]
 *
 *   const auto& delta = dlse *
 *       tile(moddims(fccgacc(span, span, t), 1, N, B), N); // [N, N, B]
 *   fccgacc(span, span, t - 1) = moddims(sum(delta, 1), N, B);
 *   gtrans += sum(delta * tile(moddims(gscale, 1, 1, B), N, N), 2);
 * }
 */

int fullConnectionCriterionBackward(
    int T,
    int B,
    int N,
    const float* trans,
    const double* fccacc,
    double* fccgacc,
    double* gtrans,
    cudaStream_t stream) {
  int retval = 0;
  double* transtmp;

  if ((retval = cudaMalloc(&transtmp, B * N * N * sizeof(double)))) {
    goto err_1;
  }

  for (int t = T - 1; t > 0; --t) {
    backwardStep1<<<B * N, kBlockSize, 0, stream>>>(
        N,
        fccacc + (t - 1) * B * N,
        fccgacc + t * B * N,
        trans,
        transtmp,
        gtrans);
    computeSums1<<<B * N, kBlockSize, 0, stream>>>(
        N, N, transtmp, fccgacc + (t - 1) * B * N);
  }

  cudaFree(transtmp);
err_1:
  return retval;
}

} // namespace cuda
} // namespace w2l

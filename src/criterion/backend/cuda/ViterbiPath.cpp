#include "criterion/CriterionUtils.h"

#include <vector>

#include <flashlight/common/cuda.h>

#include "criterion/backend/cuda/kernels/ViterbiPath.cuh"

namespace w2l {

af::array viterbiPath(const af::array& input, const af::array& trans) {
  if (input.isempty()) {
    return af::array();
  }
  auto N = input.dims(0);
  auto T = input.dims(1);
  auto B = input.dims(2);

  af::array alpha = w2l::reorder(input, 0, 2, 1); // [N, B, T]
  af::array beta(N, B, T, s32);

  {
    fl::DevicePtr trans_raw(trans);
    fl::DevicePtr alpha_raw(alpha);
    fl::DevicePtr beta_raw(beta);

    if (w2l::cuda::viterbiPath(
            T,
            B,
            N,
            static_cast<const float*>(trans_raw.get()),
            static_cast<float*>(alpha_raw.get()),
            static_cast<int*>(beta_raw.get()),
            fl::cuda::getActiveStream())) {
      throw std::runtime_error("viterbiPath failed");
    }
  }

  alpha = w2l::reorder(alpha, 0, 2, 1); // [N, T, B]
  beta = w2l::reorder(beta, 0, 2, 1); // [N, T, B]

  std::vector<float> alphaVec(B * T * N);
  std::vector<int> betaVec(B * T * N);
  std::vector<int> res(B * T);

  alpha.host(alphaVec.data());
  beta.host(betaVec.data());

  auto computeRes = [T, N](const float* alpha, const int* beta, int* res) {
    float max = NEG_INFINITY_FLT;
    int pos = -1;
    for (int i = 0; i < N; i++) {
      int k = (T - 1) * N + i;
      if (max < alpha[k]) {
        max = alpha[k];
        pos = i;
      }
    }
    res[T - 1] = pos;
    for (int t = T - 1; t > 0; t--) {
      pos = beta[t * N + pos];
      res[t - 1] = pos;
    }
  };

  for (int b = 0; b < B; ++b) {
    computeRes(&alphaVec[b * T * N], &betaVec[b * T * N], &res[b * T]);
  }

  return af::array(T, B, res.data());
}

} // namespace w2l

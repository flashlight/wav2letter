#include "criterion/CriterionUtils.h"

#include <algorithm>
#include <vector>

namespace w2l {

af::array viterbiPath(const af::array& input, const af::array& trans) {
  if (input.isempty()) {
    return af::array();
  }
  auto N = input.dims(0);
  auto T = input.dims(1);
  auto B = input.dims(2);
  std::vector<float> inputRaw(N * T * B);
  std::vector<float> transRaw(N * N);
  std::vector<int> res(T * B);

  input.host(inputRaw.data());
  trans.host(transRaw.data());

  std::vector<float> alpha(N * T);
  std::vector<int> beta(N * T);

  for (int b = 0; b < B; ++b) {
    std::copy(
        inputRaw.begin() + b * N * T,
        inputRaw.begin() + b * N * T + N,
        alpha.begin());

    for (int t = 1; t < T; t++) {
      float* alphaCurFrame = alpha.data() + t * N;
      float* alphaPrevFrame = alpha.data() + (t - 1) * N;
      float* inputCurFrame = inputRaw.data() + t * N + b * N * T;
      int* betaCurFrame = beta.data() + t * N;

      for (int i = 0; i < N; i++) {
        float max = NEG_INFINITY_FLT;
        for (int j = 0; j < N; j++) {
          float z = alphaPrevFrame[j] + transRaw[i * N + j];
          if (max < z) {
            betaCurFrame[i] = j;
            max = z;
          }
        }
        alphaCurFrame[i] = max + inputCurFrame[i];
      }
    }

    float max = NEG_INFINITY_FLT;
    float* alphaCurFrame = alpha.data() + (T - 1) * N;
    int pos = -1;
    for (int i = 0; i < N; i++) {
      if (max < alphaCurFrame[i]) {
        max = alphaCurFrame[i];
        pos = i;
      }
    }
    res[b * T + T - 1] = pos;
    for (int i = T - 1; i > 0; i--) {
      pos = beta[i * N + pos];
      res[b * T + i - 1] = pos;
    }
  }

  return af::array(T, B, res.data());
}

} // namespace w2l

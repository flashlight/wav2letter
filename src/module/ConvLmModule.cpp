/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "module/ConvLmModule.h"

#include <string>

#include "common/FlashlightUtils.h"

using namespace fl;

namespace w2l {
GetConvLmScoreFunc buildGetConvLmScoreFunction(
    std::shared_ptr<fl::Module> network) {
  auto getConvLmScoreFunc = [network](
                                const std::vector<int>& inputs,
                                const std::vector<int>& lastTokenPositions,
                                int sampleSize = -1,
                                int batchSize = 1) {
    sampleSize = sampleSize > 0 ? sampleSize : inputs.size();
    if (sampleSize * batchSize > inputs.size()) {
      throw std::invalid_argument(
          "[ConvLM] Incorrect sample size (" + std::to_string(sampleSize) +
          ") or batch size (" + std::to_string(batchSize) + ").");
    }
    af::array inputData(sampleSize, batchSize, inputs.data());
    fl::Variable output = network->forward({fl::input(inputData)})[0];

    if (af::count<int>(af::isNaN(output.array())) != 0) {
      throw std::runtime_error("[ConvLM] Encountered NaNs in propagation");
    }
    std::vector<std::vector<float>> chosenFramePred(batchSize);
    auto preds = af::reorder(output.array(), 2, 1, 0); // (b t c)
    if (preds.dims(0) != batchSize) {
      throw std::logic_error(
          "[ConvLM]: incorrect predictions: batch should be " +
          std::to_string(batchSize) + " but it is " +
          std::to_string(preds.dims(0)));
    }
    for (int idx = 0; idx < batchSize; idx++) {
      if ((lastTokenPositions[idx] < 0) ||
          (lastTokenPositions[idx] >= preds.dims(1))) {
        throw std::logic_error(
            "[ConvLM]: trying the access to batch idx " + std::to_string(idx) +
            " and time idx " + std::to_string(lastTokenPositions[idx]) +
            " while the sizes are b: " + std::to_string(preds.dims(0)) +
            " t: " + std::to_string(preds.dims(1)));
      }
      chosenFramePred[idx] =
          afToVector<float>(preds.row(idx).col(lastTokenPositions[idx]));
    }
    return chosenFramePred;
  };

  return getConvLmScoreFunc;
}
} // namespace w2l

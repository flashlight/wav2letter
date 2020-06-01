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
    int32_t C = output.dims(0), T = output.dims(1), B = output.dims(2);
    if (B != batchSize) {
      throw std::logic_error(
          "[ConvLM]: incorrect predictions: batch should be " +
          std::to_string(batchSize) + " but it is " + std::to_string(B));
    }
    if (batchSize != (int)lastTokenPositions.size()) {
      throw std::logic_error(
          "[ConvLM]: incorrect postions for accessing: size should be " +
          std::to_string(batchSize) + " but it is " +
          std::to_string(lastTokenPositions.size()));
    }
    // output (c, t, b)
    // set global indices: offset by channel
    af::array globalIndices = af::iota(af::dim4(C, 1), af::dim4(1, B), s32);
    // set global indices: offset by batch
    globalIndices =
        globalIndices + af::iota(af::dim4(1, B), af::dim4(C, 1), s32) * T * C;
    // set global indices: offset by time which we need to take
    globalIndices = globalIndices +
        af::tile(af::array(af::dim4(1, B), lastTokenPositions.data()), C, 1) *
            C;
    af::array preds =
        af::moddims(af::flat(output.array())(af::flat(globalIndices)), C, B);
    // vector of B X C predictions
    return afToVector<float>(preds);
  };

  return getConvLmScoreFunc;
}
} // namespace w2l

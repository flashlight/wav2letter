/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/polymorphic.hpp>
#include <gtest/gtest.h>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <streambuf>
#include <vector>

#include "inference/module/feature/LogMelFeature.h"
#include "inference/module/test/TestUtils.h"

namespace w2l {
namespace streaming {

TEST(LogMelFeature, LogMelFeatureTest) {
  int randomLargeValue = 166960;
  std::vector<float> inputData = randVec<float>(randomLargeValue);
  auto streamingOutput = [](const size_t chunkSize, std::vector<float>& in) {
    auto mfsclib = LogMelFeature(80 /* filter dim */);
    auto input = std::make_shared<ModuleProcessingState>(1);
    std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);

    int consumed = 0;

    auto totalSize = in.size();
    size_t curChunkSize = 0;
    float* curInPtr = in.data();
    mfsclib.start(input);
    while (consumed < totalSize) {
      curChunkSize = std::min(chunkSize, totalSize - consumed);

      inputBuffer->write(curInPtr + consumed, curChunkSize);
      consumed += curChunkSize;
      mfsclib.run(input);
    }
    auto output = mfsclib.finish(input);
    assert(output->buffers().size() == 1);
    auto outputBuf = output->buffer(0);
    assert(outputBuf);
    return std::vector<float>(
        outputBuf->data<float>(), outputBuf->tail<float>());
  };

  auto chunk50msOutput = streamingOutput(50 * 16, inputData);
  auto chunk200msOutput = streamingOutput(200 * 16, inputData);
  auto chunkAllOutput = streamingOutput(inputData.size(), inputData);

  // Verifying output is sample regardless of chunk size
  ASSERT_EQ(chunk50msOutput.size(), chunk200msOutput.size());
  ASSERT_EQ(chunk50msOutput.size(), chunkAllOutput.size());
  for (size_t i = 0; i < chunk50msOutput.size(); ++i) {
    ASSERT_NEAR(chunk50msOutput[i], chunk200msOutput[i], 1E-3);
    ASSERT_NEAR(chunk50msOutput[i], chunkAllOutput[i], 1E-3);
  }
}
} // namespace streaming
} // namespace w2l

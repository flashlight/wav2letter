/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cereal/archives/portable_binary.hpp>
#include <gtest/gtest.h>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "inference/common/DataType.h"
#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/Conv1d.h"
#include "inference/module/nn/Residual.h"
#include "inference/module/test/TestUtils.h"

namespace w2l {
namespace streaming {

TEST(Residual, ConvResidual) {
  const int T = 10, groups = 5, channels = 10;
  const std::vector<float> inputValues(T * channels, 1.0);
  const int kernelSize = 3;
  const int stride = 1;
  const int rightPadding = 1;
  const int leftPadding = 1;

  // kernelSize x 1 x channels x channels
  std::vector<float> weightsValues =
      randVec<float>(kernelSize * channels * channels / groups);
  const std::vector<float> biasValues = randVec<float>(channels / groups);

  const auto weights = std::make_shared<ModuleParameter>(
      DataType::FLOAT, weightsValues.data(), weightsValues.size());
  const auto bias = std::make_shared<ModuleParameter>(
      DataType::FLOAT, biasValues.data(), biasValues.size());

  std::shared_ptr<Conv1d> conv = createConv1d(
      channels,
      channels,
      kernelSize,
      stride,
      {leftPadding, rightPadding},
      groups,
      weights,
      bias);

  auto inputNoRes = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBufNoRes = inputNoRes->buffer(0);
  inputBufNoRes->write<float>(inputValues.data(), inputValues.size());
  conv->start(inputNoRes);
  auto outputNoRes = conv->run(inputNoRes);
  conv->finish(inputNoRes);

  auto inputRes = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBufRes = inputRes->buffer(0);
  inputBufRes->write<float>(inputValues.data(), inputValues.size());

  std::shared_ptr<Residual> residual =
      std::make_shared<Residual>(conv, DataType::FLOAT);

  residual->start(inputRes);
  auto outputRes = residual->run(inputRes);
  residual->finish(inputRes);

  std::shared_ptr<IOBuffer> outputBufNoRes = outputNoRes->buffer(0);
  std::shared_ptr<IOBuffer> outputBufRes = outputRes->buffer(0);
  float* outNoRes = outputBufNoRes->data<float>();
  float* outRes = outputBufRes->data<float>();
  ASSERT_EQ(outputBufNoRes->size<float>(), outputBufRes->size<float>());
  for (int i = 0; i < outputBufNoRes->size<float>(); ++i) {
    ASSERT_NEAR(outNoRes[i] + 1.0, outRes[i], 1E-3);
  }
}

TEST(Residual, ConvResidualSerialization) {
  const int T = 10, groups = 5, channels = 10;
  const std::vector<float> inputValues(T * channels, 1.0);
  const int kernelSize = 3;
  const int stride = 1;
  const int rightPadding = 1;
  const int leftPadding = 1;

  std::stringstream memoryBufferStream;
  std::shared_ptr<ModuleProcessingState> outputNoRes;
  {
    // kernelSize x 1 x channels x channels
    std::vector<float> weightsValues =
        randVec<float>(kernelSize * channels * channels / groups);
    const std::vector<float> biasValues = randVec<float>(channels / groups);

    const auto weights = std::make_shared<ModuleParameter>(
        DataType::FLOAT, weightsValues.data(), weightsValues.size());
    const auto bias = std::make_shared<ModuleParameter>(
        DataType::FLOAT, biasValues.data(), biasValues.size());

    std::shared_ptr<Conv1d> conv = createConv1d(
        channels,
        channels,
        kernelSize,
        stride,
        {leftPadding, rightPadding},
        groups,
        weights,
        bias);

    auto inputNoRes = std::make_shared<ModuleProcessingState>(1);
    std::shared_ptr<IOBuffer> inputBufNoRes = inputNoRes->buffer(0);
    inputBufNoRes->write<float>(inputValues.data(), inputValues.size());
    conv->start(inputNoRes);
    outputNoRes = conv->run(inputNoRes);
    conv->finish(inputNoRes);

    std::shared_ptr<Residual> residual_ =
        std::make_shared<Residual>(conv, DataType::FLOAT);

    std::cout << "Before serialization:" << residual_->debugString()
              << std::endl;
    cereal::BinaryOutputArchive archive(memoryBufferStream);
    archive(residual_);
  }

  std::shared_ptr<Residual> residual;
  {
    cereal::BinaryInputArchive archive(memoryBufferStream);
    archive(residual);
  }
  std::cout << "After serialization:" << residual->debugString() << std::endl;

  auto inputRes = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBufRes = inputRes->buffer(0);
  inputBufRes->write<float>(inputValues.data(), inputValues.size());

  residual->start(inputRes);
  auto outputRes = residual->run(inputRes);
  residual->finish(inputRes);

  std::shared_ptr<IOBuffer> outputBufNoRes = outputNoRes->buffer(0);
  std::shared_ptr<IOBuffer> outputBufRes = outputRes->buffer(0);
  float* outNoRes = outputBufNoRes->data<float>();
  float* outRes = outputBufRes->data<float>();
  ASSERT_EQ(outputBufNoRes->size<float>(), outputBufRes->size<float>());
  for (int i = 0; i < outputBufNoRes->size<float>(); ++i) {
    ASSERT_NEAR(outNoRes[i] + 1.0, outRes[i], 1E-3);
  }
}

} // namespace streaming
} // namespace w2l

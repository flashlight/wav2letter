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
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <streambuf>
#include <vector>

#include "inference/common/DataType.h"
#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/Linear.h"
#include "inference/module/nn/backend/fbgemm/LinearFbGemm.h"

namespace w2l {
namespace streaming {

TEST(Linear, SingleNeuronSingleFrame) {
  const std::vector<float> weightsValues = {0.1, 1.0, 10};
  const std::vector<float> biasValues = {7.9};

  const auto weights = std::make_shared<ModuleParameter>(
      DataType::FLOAT, weightsValues.data(), weightsValues.size());
  const auto bias = std::make_shared<ModuleParameter>(
      DataType::FLOAT, biasValues.data(), biasValues.size());

  std::shared_ptr<Linear> lin = createLinear(3, 1, weights, bias);
  std::cout << "TEST(Linear, SingleNeuronSingleFrame)\n"
            << "Linear:" << lin->debugString() << std::endl;

  const std::vector<float> inputValues = {1.0, 2.0, 3.0};
  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size());

  lin->start(input);
  auto output = lin->run(input);
  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  float* out = outputBuffer->data<float>();
  for (int i = 0; i < outputBuffer->size<float>(); ++i) {
    std::cout << i << "=" << out[i] << std::endl;
  }
  ASSERT_EQ(outputBuffer->size<float>(), 1);
  ASSERT_NEAR(out[0], 40.0, 1E-3);
}

TEST(Linear, MultiNeuronMultiFrame) {
  const std::vector<float> weightsValues = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const std::vector<float> biasValues = {2.0, 0.0};

  const auto weights = std::make_shared<ModuleParameter>(
      DataType::FLOAT, weightsValues.data(), weightsValues.size());
  const auto bias = std::make_shared<ModuleParameter>(
      DataType::FLOAT, biasValues.data(), biasValues.size());

  std::shared_ptr<Linear> lin = createLinear(3, 2, weights, bias);

  const std::vector<float> inputValues = {1.0, 1.0, 1.0, 10.0, 10.0, 10.0};
  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size());
  lin->start(input);
  auto output = lin->run(input);
  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  float* out = outputBuffer->data<float>();
  ASSERT_EQ(outputBuffer->size<float>(), 4);
  ASSERT_NEAR(out[0], 5.0, 1E-3);
  ASSERT_NEAR(out[1], 3.0, 1E-3);
  ASSERT_NEAR(out[2], 32.0, 1E-3);
  ASSERT_NEAR(out[3], 30.0, 1E-3);
}

// Same test as MultiNeuronMultiFrame but with serialization.
TEST(Linear, Serialization) {
  std::stringstream memoryBufferStream;
  {
    const std::vector<float> weightsValues = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const std::vector<float> biasValues = {2.0, 0.0};

    const auto weights = std::make_shared<ModuleParameter>(
        DataType::FLOAT, weightsValues.data(), weightsValues.size());
    const auto bias = std::make_shared<ModuleParameter>(
        DataType::FLOAT, biasValues.data(), biasValues.size());

    std::shared_ptr<Linear> lin = createLinear(3, 2, weights, bias);
    std::cout << "TEST(Linear, Serialization)\n"
              << "Before serialization:" << lin->debugString() << std::endl;

    cereal::BinaryOutputArchive archive(memoryBufferStream);
    archive(lin);
  }

  std::shared_ptr<Linear> lin;
  cereal::BinaryInputArchive archive(memoryBufferStream);
  archive(lin);
  std::cout << "TEST(Linear, Serialization)\n"
            << "After serialization:" << lin->debugString() << std::endl;

  const std::vector<float> inputValues = {1.0, 1.0, 1.0, 10.0, 10.0, 10.0};
  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size());
  lin->start(input);
  auto output = lin->run(input);
  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  float* out = outputBuffer->data<float>();
  ASSERT_EQ(outputBuffer->size<float>(), 4);
  ASSERT_NEAR(out[0], 5.0, 1E-3);
  ASSERT_NEAR(out[1], 3.0, 1E-3);
  ASSERT_NEAR(out[2], 32.0, 1E-3);
  ASSERT_NEAR(out[3], 30.0, 1E-3);
}
} // namespace streaming
} // namespace w2l

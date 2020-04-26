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

#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/LayerNorm.h"
#include "inference/module/test/TestUtils.h"

namespace w2l {
namespace streaming {

TEST(LayerNorm, Batch) {
  int T = 100, F = 1000;
  auto mean = randVec<float>(T);
  auto std = randVec<float>(T);
  float alpha = 0.2, beta = 0.1;
  std::vector<float> inputValues(T * F);
  for (int i = 0; i < T; ++i) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean[i], std[i]);
    for (int j = 0; j < F; ++j) {
      inputValues[i * F + j] = distribution(generator);
    }
  }

  LayerNorm layerNorm(F, alpha, beta);
  std::cout << "TEST(LayerNorm, Batch)\n"
            << "LayerNorm:" << layerNorm.debugString() << std::endl;

  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size());

  layerNorm.start(input);
  auto output = layerNorm.run(input);
  layerNorm.finish(input);

  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  ASSERT_EQ(inputBuffer->size<char>(), 0);
  ASSERT_EQ(outputBuffer->size<float>(), inputValues.size());

  float* outPtr = outputBuffer->data<float>();
  float* inPtr = inputValues.data();
  for (int i = 0; i < T; ++i) {
    for (int j = 0; j < F; ++j) {
      auto e = alpha * ((inPtr[i * F + j] - mean[i]) / std[i]) + beta;
      EXPECT_NEAR(outPtr[i * F + j], e, 1e-1);
    }
  }
}

TEST(LayerNorm, BatchChunked) {
  int T = 100, F = 1000;
  auto mean = randVec<float>(T);
  auto std = randVec<float>(T);
  float alpha = 0.2, beta = 0.1;
  std::vector<float> inputValues(T * F);
  for (int i = 0; i < T; ++i) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean[i], std[i]);
    for (int j = 0; j < F; ++j) {
      inputValues[i * F + j] = distribution(generator);
    }
  }

  LayerNorm layerNorm(F, alpha, beta);
  std::cout << "TEST(LayerNorm, BatchChunked)\n"
            << "LayerNorm:" << layerNorm.debugString() << std::endl;

  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);

  auto output = layerNorm.start(input);
  int chunkSize = inputValues.size() / 10;
  for (int i = 0; i < 10; ++i) {
    inputBuffer->write<float>(inputValues.data() + i * chunkSize, chunkSize);
    layerNorm.run(input);
  }
  layerNorm.finish(input);

  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  ASSERT_EQ(inputBuffer->size<char>(), 0);
  ASSERT_EQ(outputBuffer->size<float>(), inputValues.size());

  float* outPtr = outputBuffer->data<float>();
  float* inPtr = inputValues.data();
  for (int i = 0; i < T; ++i) {
    for (int j = 0; j < F; ++j) {
      auto e = alpha * ((inPtr[i * F + j] - mean[i]) / std[i]) + beta;
      ASSERT_NEAR(outPtr[i * F + j], e, 1e-1);
    }
  }
}

} // namespace streaming
} // namespace w2l

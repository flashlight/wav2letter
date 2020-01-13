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

#include "inference/common/DataType.h"
#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/Relu.h"

namespace w2l {
namespace streaming {

TEST(Relu, basic) {
  Relu relu(DataType::FLOAT);
  std::cout << "TEST(Relu, basic)\n"
            << "relu:" << relu.debugString() << std::endl;

  const std::vector<float> inputValues = {
      1.0, -1.0, 0.0, 10.01, -10.01, static_cast<float>(INT_MAX)};
  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size());

  relu.start(input);
  auto output = relu.run(input);
  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  ASSERT_EQ(outputBuffer->size<float>(), 6);
  float* out = outputBuffer->data<float>();
  ASSERT_NEAR(out[0], 1.0, 1E-3);
  ASSERT_NEAR(out[1], 0.0, 1E-3);
  ASSERT_NEAR(out[2], 0.0, 1E-3);
  ASSERT_NEAR(out[3], 10.01, 1E-3);
  ASSERT_NEAR(out[4], 0.0, 1E-3);
  ASSERT_NEAR(out[5], INT_MAX, 1E+1);
}
} // namespace streaming
} // namespace w2l

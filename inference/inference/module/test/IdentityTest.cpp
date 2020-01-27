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
#include <numeric>
#include <random>
#include <streambuf>
#include <vector>

#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/Identity.h"

namespace w2l {
namespace streaming {

TEST(Identity, SimpleCopy) {
  std::shared_ptr<Identity> copy = std::make_shared<Identity>();

  std::vector<float> inputValues = {
      0.87, 0.32, 0.62, 0.28, 0.48, 0.20, 0.20, 0.71, 0.87, 0.40};
  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size() / 2);

  copy->start(input);
  copy->run(input);
  inputBuffer->write<float>(
      inputValues.data() + inputValues.size() / 2, inputValues.size() / 2);
  copy->run(input);
  auto output = copy->finish(input);

  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);
  float* out = outputBuffer->data<float>();
  for (int i = 0; i < outputBuffer->size<float>(); ++i) {
    ASSERT_NEAR(out[i], inputValues[i], 1E-4);
  }
}
} // namespace streaming
} // namespace w2l

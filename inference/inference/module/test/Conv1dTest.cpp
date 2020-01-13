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
#include <utility>
#include <vector>

#include "inference/common/DataType.h"
#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/Conv1d.h"
#include "inference/module/nn/Relu.h"

namespace w2l {
namespace streaming {

TEST(Conv1d, SingleLayer) {
  // clang-format off
    const int T = 10, groups = 5, channels = 10, B = 1;
    // T x groups x (channels/groups) x B
    const std::vector<float> inputValues = {
      0.601, 0.909, 0.158, 0.597, 0.675, 0.904, 0.514, 0.396, 0.791, 0.889,
      0.02776, 0.8865, 0.3712, 0.9594, 0.6105, 0.01985, 0.367, 0.8055, 0.1654, 0.3434,
      0.9806, 0.9676, 0.3543, 0.2323, 0.5232, 0.4436, 0.3336, 0.5147, 0.8657, 0.804,
      0.2126, 0.1425, 0.645, 0.9623, 0.5567, 0.6808, 0.03628, 0.321, 0.3766, 0.2986,
      0.06546, 0.5137, 0.9675, 0.8578, 0.7896, 0.6636, 0.5349, 0.7831, 0.7331, 0.04377,
      0.5497, 0.6484, 0.3636, 0.01921, 0.8966, 0.8738, 0.01227, 0.4784, 0.2522, 0.6108,
      0.2864, 0.6353, 0.4165, 0.7191, 0.05364, 0.3954, 0.3988, 0.577, 0.9644, 0.9033,
      0.341, 0.7449, 0.5814, 0.4035, 0.5775, 0.5277, 0.9787, 0.8173, 0.4711, 0.5131,
      0.7509, 0.4391, 0.8962, 0.4692, 0.2908, 0.3592, 0.2308, 0.5207, 0.3637, 0.5784,
      0.4105, 0.6982, 0.3712, 0.3353, 0.9941, 0.8567, 0.6244, 0.4341, 0.9643, 0.691};

    // T x groups x (channels/groups) x B
    const std::vector<float> targetValues = {
      1.378, 1.373, 1.199, 1.453, 1.535, 1.446, 1.246, 1.417, 1.389, 1.294,
      2.802, 2.828, 1.85, 1.678, 2.164, 2.298, 1.845, 2.004, 2.501, 2.757,
      2.283, 1.665, 2.308, 2.3, 1.683, 2.112, 1.642, 1.657, 1.897, 1.7,
      1.874, 2.285, 2.488, 2.442, 2.327, 2.4, 1.616, 1.911, 2.194, 2.454,
      1.334, 1.671, 2.638, 2.315, 2.837, 2.718, 1.393, 1.314, 1.456, 1.7,
      1.762, 1.625, 2.131, 2.545, 2.326, 2.272, 1.842, 2.072, 2.072, 2.58,
      2.017, 2.163, 1.574, 1.829, 2.185, 2.545, 2.148, 2.039, 2.434, 2.075,
      2.168, 2.098, 2.372, 2.328, 1.505, 1.442, 2.26, 2.029, 2.433, 2.621,
      2.21, 2.075, 1.981, 2.057, 2.292, 2.553, 2.36, 2.671, 2.32, 2.433,
      1.525, 1.813, 1.384, 1.864, 1.694, 1.504, 1.325, 1.316, 1.801, 1.607
    };

    const int kernelSize = 3;
    const int stride = 1;
    const int rightPadding=1;
    const int leftPadding=1;

    // kernelSize x 1 x channels x channels
    std::vector<float> weightsValues = {
      0.4792, 0.8523, 0.6265, 0.5827, 0.7387, 0.3657,
      0.973, 0.3688, 0.2184, 0.2578, 0.6659, 0.3815
    };
    const std::vector<float> biasValues = {0.1266, 0.6511};
  // clang-format on

  const auto weights = std::make_shared<ModuleParameter>(
      DataType::FLOAT, weightsValues.data(), weightsValues.size());
  const auto bias = std::make_shared<ModuleParameter>(
      DataType::FLOAT, biasValues.data(), biasValues.size());

  std::shared_ptr<Conv1d> conv = createConv1d(
      channels,
      channels,
      kernelSize,
      stride,
      std::make_pair(leftPadding, rightPadding),
      groups,
      weights,
      bias);

  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size());

  conv->start(input);
  auto output = conv->run(input);
  conv->finish(input);

  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  ASSERT_EQ(outputBuffer->size<float>(), T * channels * B);
  float* out = outputBuffer->data<float>();
  for (int i = 0; i < T * channels * B; i++) {
    std::cout << i << " " << out[i] << " " << targetValues[i] << std::endl;
    ASSERT_NEAR(out[i], targetValues[i], 1E-2);
  }
}

TEST(Conv1d, SingleLayerSerialization) {
  const int T = 10, groups = 5, channels = 10, B = 1;
  std::stringstream memoryBufferStream;
  {
    const int kernelSize = 3;
    const int stride = 1;
    const int rightPadding = 1;
    const int leftPadding = 1;

    // kernelSize x 1 x channels x channels
    std::vector<float> weightsValues = {0.4792,
                                        0.8523,
                                        0.6265,
                                        0.5827,
                                        0.7387,
                                        0.3657,
                                        0.973,
                                        0.3688,
                                        0.2184,
                                        0.2578,
                                        0.6659,
                                        0.3815};
    const std::vector<float> biasValues = {0.1266, 0.6511};
    // clang-format on

    const auto weights = std::make_shared<ModuleParameter>(
        DataType::FLOAT, weightsValues.data(), weightsValues.size());
    const auto bias = std::make_shared<ModuleParameter>(
        DataType::FLOAT, biasValues.data(), biasValues.size());

    std::shared_ptr<Conv1d> conv = createConv1d(
        channels,
        channels,
        kernelSize,
        stride,
        std::make_pair(leftPadding, rightPadding),
        groups,
        weights,
        bias);

    std::cout << "TEST(Conv1d, SingleLAyerSerialization)\n"
              << "Before serialization:" << conv->debugString() << std::endl;

    cereal::BinaryOutputArchive archive(memoryBufferStream);
    archive(conv);
  }

  std::shared_ptr<Conv1d> conv;
  cereal::BinaryInputArchive archive(memoryBufferStream);
  archive(conv);

  std::cout << "TEST(Conv1d, SingleLAyerSerialization)\n"
            << "After serialization:" << conv->debugString() << std::endl;

  // clang-format off
    // T x groups x channels x B
    const std::vector<float> inputValues = {
      0.601, 0.909, 0.158, 0.597, 0.675, 0.904, 0.514, 0.396, 0.791, 0.889,
      0.02776, 0.8865, 0.3712, 0.9594, 0.6105, 0.01985, 0.367, 0.8055, 0.1654, 0.3434,
      0.9806, 0.9676, 0.3543, 0.2323, 0.5232, 0.4436, 0.3336, 0.5147, 0.8657, 0.804,
      0.2126, 0.1425, 0.645, 0.9623, 0.5567, 0.6808, 0.03628, 0.321, 0.3766, 0.2986,
      0.06546, 0.5137, 0.9675, 0.8578, 0.7896, 0.6636, 0.5349, 0.7831, 0.7331, 0.04377,
      0.5497, 0.6484, 0.3636, 0.01921, 0.8966, 0.8738, 0.01227, 0.4784, 0.2522, 0.6108,
      0.2864, 0.6353, 0.4165, 0.7191, 0.05364, 0.3954, 0.3988, 0.577, 0.9644, 0.9033,
      0.341, 0.7449, 0.5814, 0.4035, 0.5775, 0.5277, 0.9787, 0.8173, 0.4711, 0.5131,
      0.7509, 0.4391, 0.8962, 0.4692, 0.2908, 0.3592, 0.2308, 0.5207, 0.3637, 0.5784,
      0.4105, 0.6982, 0.3712, 0.3353, 0.9941, 0.8567, 0.6244, 0.4341, 0.9643, 0.691};

    // T x groups x channels x B
    const std::vector<float> targetValues = {
      1.378, 1.373, 1.199, 1.453, 1.535, 1.446, 1.246, 1.417, 1.389, 1.294,
      2.802, 2.828, 1.85, 1.678, 2.164, 2.298, 1.845, 2.004, 2.501, 2.757,
      2.283, 1.665, 2.308, 2.3, 1.683, 2.112, 1.642, 1.657, 1.897, 1.7,
      1.874, 2.285, 2.488, 2.442, 2.327, 2.4, 1.616, 1.911, 2.194, 2.454,
      1.334, 1.671, 2.638, 2.315, 2.837, 2.718, 1.393, 1.314, 1.456, 1.7,
      1.762, 1.625, 2.131, 2.545, 2.326, 2.272, 1.842, 2.072, 2.072, 2.58,
      2.017, 2.163, 1.574, 1.829, 2.185, 2.545, 2.148, 2.039, 2.434, 2.075,
      2.168, 2.098, 2.372, 2.328, 1.505, 1.442, 2.26, 2.029, 2.433, 2.621,
      2.21, 2.075, 1.981, 2.057, 2.292, 2.553, 2.36, 2.671, 2.32, 2.433,
      1.525, 1.813, 1.384, 1.864, 1.694, 1.504, 1.325, 1.316, 1.801, 1.607
    };

  auto input = std::make_shared<ModuleProcessingState>(1);
  std::shared_ptr<IOBuffer> inputBuffer = input->buffer(0);
  inputBuffer->write<float>(inputValues.data(), inputValues.size());

  conv->start(input);
  auto output = conv->run(input);
  conv->finish(input);

  std::shared_ptr<IOBuffer> outputBuffer = output->buffer(0);

  ASSERT_EQ(outputBuffer->size<float>(), T * channels * B);
  float* out = outputBuffer->data<float>();
  for (int i = 0; i < T * channels * B; i++) {
    ASSERT_NEAR(out[i], targetValues[i], 1E-2);
  }
}
}
}

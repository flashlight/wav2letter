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
#include <numeric>
#include <streambuf>
#include <vector>

#include "inference/common/DataType.h"
#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/TDSBlock.h"
#include "inference/module/test/TestUtils.h"

namespace w2l {
namespace streaming {

TEST(TDSBlock, TestOne) {
  const int T = 10, groups = 5, channels = 10;
  const std::vector<float> inputValues(T * channels, 1.0);
  const int kernelSize = 3;
  const int stride = 1;
  const int rightPadding = 1;
  const int leftPadding = 1;
  std::vector<float> conv_weights = {-0.02946,
                                     0.4982,
                                     0.1789,
                                     0.117,
                                     0.3376,
                                     -0.1899,
                                     0.6689,
                                     -0.1856,
                                     -0.3983,
                                     -0.3425,
                                     0.2346,
                                     -0.1675};
  std::vector<float> conv_bias = {-0.3049, 0.1234};
  std::vector<float> ln1_weights = {1};
  std::vector<float> ln1_bias = {0};
  std::vector<float> lin1_weights = {
      -0.05403, 0.3335,   -0.08549, 0.34,     -0.2749, 0.2095,   -0.09552,
      0.309,    0.2918,   -0.2132,  0.4874,   0.2073,  0.09121,  -0.4156,
      0.2846,   -0.3428,  0.06689,  0.06052,  0.1128,  -0.2831,  -0.3645,
      -0.03331, -0.2202,  -0.2175,  -0.2408,  0.2123,  -0.3856,  -0.1629,
      -0.402,   0.2384,   0.135,    -0.3414,  -0.1583, -0.5424,  -0.08242,
      -0.1408,  -0.3339,  -0.1275,  -0.4359,  0.105,   -0.4183,  -0.1536,
      0.3727,   -0.07812, 0.4965,   -0.1655,  0.4306,  0.4748,   0.1286,
      0.3827,   -0.51,    -0.5359,  -0.09174, -0.5288, -0.01354, 0.05352,
      -0.5216,  0.2862,   -0.4435,  -0.2302,  0.01024, -0.1423,  0.02748,
      -0.3257,  0.1114,   -0.2293,  -0.3592,  -0.4782, -0.3439,  0.4165,
      0.03735,  -0.1461,  -0.5157,  0.3248,   -0.3211, 0.4939,   -0.4959,
      0.4564,   -0.517,   -0.2229,  0.4333,   0.04413, -0.4921,  0.261,
      0.3395,   0.2823,   0.01454,  -0.5391,  0.4163,  -0.3513,  0.5264,
      0.3795,   -0.2339,  0.4819,   -0.1151,  0.1974,  -0.1577,  -0.1415,
      0.1202,   -0.2464};

  std::vector<float> lin1_bias = {-0.1442,
                                  -0.1367,
                                  -0.2478,
                                  -0.02559,
                                  -0.2541,
                                  0.273,
                                  0.2599,
                                  0.00358,
                                  -0.1015,
                                  0.1126};
  std::vector<float> lin2_weights = {
      0.5176,   -0.03139, -0.2531,  -0.3564, 0.02125,   -0.07353, -0.4867,
      -0.3793,  0.3697,   -0.4386,  0.4392,  -0.002563, 0.1399,   0.3891,
      -0.2269,  0.3968,   -0.1685,  0.4153,  0.2734,    0.5078,   -0.1909,
      0.3533,   -0.2293,  -0.08992, 0.02778, 0.5292,    0.07255,  -0.3796,
      0.1404,   0.2538,   -0.0387,  0.07189, -0.08403,  0.1214,   0.2038,
      0.06606,  -0.4281,  -0.2828,  0.2175,  0.4887,    -0.5297,  0.4885,
      0.04326,  0.05605,  -0.1057,  0.5248,  -0.1697,   0.1174,   0.1132,
      0.3184,   -0.5145,  -0.3841,  -0.2317, 0.01497,   0.1566,   -0.4326,
      -0.07945, -0.02592, -0.157,   0.3612,  0.4875,    -0.09626, -0.1351,
      0.1591,   -0.2849,  -0.04782, -0.191,  0.5392,    0.4791,   -0.06339,
      -0.1775,  0.1179,   -0.01665, 0.3163,  -0.2075,   0.09968,  0.03366,
      -0.529,   -0.1435,  -0.3213,  0.07692, 0.07543,   0.07786,  -0.252,
      -0.2965,  -0.02569, 0.159,    0.04749, -0.1527,   -0.1368,  0.3802,
      -0.1105,  0.4154,   0.2906,   0.133,   0.455,     0.223,    0.06566,
      0.2066,   0.4582};

  std::vector<float> lin2_bias = {-0.07239,
                                  -0.03344,
                                  -0.2481,
                                  0.1835,
                                  -0.1299,
                                  0.02469,
                                  0.1474,
                                  0.09904,
                                  -0.1269,
                                  -0.1051};
  std::vector<float> ln2_weights = {1};
  std::vector<float> ln2_bias = {0};
  std::vector<float> in = {
      0.601,   0.9092,  0.1583,  0.597,  0.6755,  0.9048,  0.5143, 0.3967,
      0.7917,  0.8891,  0.02776, 0.8865, 0.3712,  0.9594,  0.6105, 0.01985,
      0.367,   0.8055,  0.1654,  0.3434, 0.9806,  0.9676,  0.3543, 0.2323,
      0.5232,  0.4436,  0.3336,  0.5147, 0.8657,  0.804,   0.2126, 0.1425,
      0.645,   0.9623,  0.5567,  0.6808, 0.03628, 0.321,   0.3766, 0.2986,
      0.06546, 0.5137,  0.9675,  0.8578, 0.7896,  0.6636,  0.5349, 0.7831,
      0.7331,  0.04377, 0.5497,  0.6484, 0.3636,  0.01921, 0.8966, 0.8738,
      0.01227, 0.4784,  0.2522,  0.6108, 0.2864,  0.6353,  0.4165, 0.7191,
      0.05364, 0.3954,  0.3988,  0.577,  0.9644,  0.9033,  0.341,  0.7449,
      0.5814,  0.4035,  0.5775,  0.5277, 0.9787,  0.8173,  0.4711, 0.5131,
      0.7509,  0.4391,  0.8962,  0.4692, 0.2908,  0.3592,  0.2308, 0.5207,
      0.3637,  0.5784,  0.4105,  0.6982, 0.3712,  0.3353,  0.9941, 0.8567,
      0.6244,  0.4341,  0.9643,  0.691};
  std::vector<float> expectedOutput = {
      0.207,   1.25,    -1.992,   -0.3494, -0.5097, 1.321,   -0.8933, -0.4819,
      1.055,   0.3938,  -0.01213, 1.315,   -0.878,  1.583,   0.2527,  -1.224,
      -1.702,  0.613,   0.3076,   -0.2551, 2.013,   0.1899,  -0.6089, -0.7738,
      -1.137,  -0.1167, -1.335,   -0.1128, 1.131,   0.75,    -1.045,  -0.5406,
      0.3942,  1.626,   0.3477,   0.1045,  -2.149,  -0.1666, 0.808,   0.6212,
      -1.475,  -0.8904, 1.989,    0.7747,  0.7267,  0.3567,  -0.3794, 0.1513,
      0.03271, -1.286,  -0.02927, 0.5477,  -0.955,  0.6344,  0.8683,  1.508,
      -1.688,  -0.8558, -0.9574,  0.9263,  -0.2974, 0.01085, -1.223,  -0.009899,
      -0.921,  0.5447,  -0.6032,  -0.5895, 2.491,   0.5972,  -0.3947, -0.7351,
      1.116,   -1.238,  -0.8152,  -0.852,  2.158,   0.2258,  -0.1106, 0.6462,
      0.4182,  -1.587,  0.9697,   0.1041,  -0.354,  -0.6139, -1.394,  0.7207,
      -0.1022, 1.838,   -1.066,   0.9314,  -1.627,  -0.2341, -0.1102, 1.134,
      -0.5934, -0.1856, 1.894,    -0.1437};

  // kernelSize x 1 x channels x channels

  const auto convWtParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, conv_weights.data(), conv_weights.size());
  const auto convBsParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, conv_bias.data(), conv_bias.size());

  const auto lin1WtParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin1_weights.data(), lin1_weights.size());
  const auto lin1BsParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin1_bias.data(), lin1_bias.size());

  const auto lin2WtParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin2_weights.data(), lin2_weights.size());
  const auto lin2BsParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin2_bias.data(), lin2_bias.size());

  auto conv = createConv1d(
      channels,
      channels,
      kernelSize,
      stride,
      {leftPadding, rightPadding},
      groups,
      convWtParam,
      convBsParam);

  auto lnorm1 =
      std::make_shared<LayerNorm>(channels, ln1_weights[0], ln1_bias[0]);
  auto lnorm2 =
      std::make_shared<LayerNorm>(channels, ln2_weights[0], ln2_bias[0]);

  auto linear1 = createLinear(channels, channels, lin1WtParam, lin1BsParam);
  auto linear2 = createLinear(channels, channels, lin2WtParam, lin2BsParam);

  auto tds = std::make_shared<TDSBlock>(
      conv, lnorm1, linear1, linear2, lnorm2, DataType::FLOAT, DataType::FLOAT);

  auto input = std::make_shared<ModuleProcessingState>(1);
  tds->start(input);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  inputBuf->write<float>(in.data(), in.size());
  auto output = tds->run(input);
  tds->finish(input);

  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  float* outPtr = outputBuf->data<float>();
  ASSERT_EQ(outputBuf->size<float>(), in.size());
  for (int i = 0; i < outputBuf->size<float>(); ++i) {
    ASSERT_NEAR(outPtr[i], expectedOutput[i], 1E-2);
  }
}

TEST(TDSBlock, Serialization) {
  const int T = 10, groups = 5, channels = 10;
  const std::vector<float> inputValues = randVec<float>(T * groups * channels);
  const int kernelSize = 3;
  const int stride = 1;
  const int rightPadding = 1;
  const int leftPadding = 1;
  int channelsPerGroup = channels / groups;
  std::vector<float> conv_weights =
      randVec<float>(kernelSize * channelsPerGroup * channelsPerGroup);
  std::vector<float> conv_bias = randVec<float>(channelsPerGroup);
  std::vector<float> ln1_weights = {1};
  std::vector<float> ln1_bias = {0};
  std::vector<float> lin1_weights =
      randVec<float>(channelsPerGroup * channelsPerGroup);
  std::vector<float> lin1_bias = randVec<float>(channels);
  std::vector<float> lin2_weights = randVec<float>(channels * channels);
  std::vector<float> lin2_bias = randVec<float>(channels);
  std::vector<float> ln2_weights = {1};
  std::vector<float> ln2_bias = {0};
  const auto convWtParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, conv_weights.data(), conv_weights.size());
  const auto convBsParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, conv_bias.data(), conv_bias.size());

  const auto lin1WtParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin1_weights.data(), lin1_weights.size());
  const auto lin1BsParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin1_bias.data(), lin1_bias.size());

  const auto lin2WtParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin2_weights.data(), lin2_weights.size());
  const auto lin2BsParam = std::make_shared<ModuleParameter>(
      DataType::FLOAT, lin2_bias.data(), lin2_bias.size());

  auto conv = createConv1d(
      channels,
      channels,
      kernelSize,
      stride,
      {leftPadding, rightPadding},
      groups,
      convWtParam,
      convBsParam);

  auto lnorm1 = std::make_shared<LayerNorm>(
      channels * groups, ln1_weights[0], ln1_bias[0]);
  auto lnorm2 = std::make_shared<LayerNorm>(
      channels * groups, ln2_weights[0], ln2_bias[0]);

  auto linear1 = createLinear(channels, channels, lin1WtParam, lin1BsParam);
  auto linear2 = createLinear(channels, channels, lin2WtParam, lin2BsParam);

  auto tds = std::make_shared<TDSBlock>(
      conv, lnorm1, linear1, linear2, lnorm2, DataType::FLOAT, DataType::FLOAT);

  auto input = std::make_shared<ModuleProcessingState>(1);
  tds->start(input);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  inputBuf->write<float>(inputValues.data(), inputValues.size());
  auto output = tds->run(input);
  tds->finish(input);
  std::cout << "Before serialization:" << tds->debugString() << std::endl;
  std::stringstream memoryBufferStream;
  {
    cereal::BinaryOutputArchive archive(memoryBufferStream);
    archive(tds);
  }

  std::shared_ptr<TDSBlock> tdsLoaded;
  {
    cereal::BinaryInputArchive archive(memoryBufferStream);
    archive(tdsLoaded);
  }
  std::cout << "After serialization:" << tdsLoaded->debugString() << std::endl;

  auto input2 = std::make_shared<ModuleProcessingState>(1);
  tdsLoaded->start(input2);
  std::shared_ptr<IOBuffer> inputBuf2 = input2->buffer(0);
  inputBuf2->write<float>(inputValues.data(), inputValues.size());
  auto output2 = tdsLoaded->run(input2);
  tdsLoaded->finish(input2);

  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  float* outPtr = outputBuf->data<float>();

  std::shared_ptr<IOBuffer> outputBuf2 = output2->buffer(0);
  float* outPtr2 = outputBuf->data<float>();

  ASSERT_EQ(outputBuf->size<float>(), outputBuf2->size<float>());
  for (int i = 0; i < outputBuf->size<float>(); ++i) {
    ASSERT_NEAR(outPtr[i], outPtr2[i], 1E-2);
  }
}
} // namespace streaming
} // namespace w2l

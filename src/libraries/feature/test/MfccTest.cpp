/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include "TestUtils.h"
#include "common/FlashlightUtils.h"
#include "libraries/feature/FeatureParams.h"
#include "libraries/feature/Mfcc.h"

namespace {
std::string loadPath = "";

auto loadData = [](const std::string& filepath) {
  std::vector<float> data;
  std::ifstream file(filepath);
  std::istream_iterator<float> eos;
  std::istream_iterator<float> iit(file);
  std::copy(iit, eos, std::back_inserter(data));
  return data;
};
} // namespace

using namespace w2l;

// HTK Code used -
//    HCopy -C config.mfcc sa1.wav sa1-mfcc.htk
// Reference : https://labrosa.ee.columbia.edu/matlab/rastamat/mfccs.html
TEST(MfccTest, htkCompareTest) {
  // read wav data
  auto wavinput = loadData(w2l::pathsConcat(loadPath, "sa1.dat"));
  ASSERT_TRUE(wavinput.size() > 0 && "sa1 frames not loaded properly!");

  // read expected output data computed from HTK
  auto htkfeat = loadData(w2l::pathsConcat(loadPath, "sa1-mfcc.htk"));
  // HTK features not read properly!
  ASSERT_TRUE(htkfeat.size() > 0);
  FeatureParams params;
  params.samplingFreq = 16000;
  params.lowFreqFilterbank = 0;
  params.highFreqFilterbank = 8000;
  params.zeroMeanFrame = true;
  params.numFilterbankChans = 20;
  params.numCepstralCoeffs = 13;
  params.useEnergy = false;
  params.zeroMeanFrame = false;
  params.usePower = false;
  Mfcc mfcc(params);
  auto feat = mfcc.apply(wavinput);
  ASSERT_EQ(feat.size(), htkfeat.size());

  ASSERT_TRUE(feat.size() % 39 == 0);
  auto numframes = feat.size() / 39;

  // HTK keeps C0 at last position. adjust accordingly.
  auto featcopy(feat);
  for (int f = 0; f < numframes; ++f) {
    for (int i = 1; i < 39; ++i) {
      feat[f * 39 + i - 1] = feat[f * 39 + i];
    }
    feat[f * 39 + 12] = featcopy[f * 39 + 0];
    feat[f * 39 + 25] = featcopy[f * 39 + 13];
    feat[f * 39 + 38] = featcopy[f * 39 + 26];
  }
  float sum = 0.0, max = 0.0;
  for (int i = 0; i < feat.size(); ++i) {
    auto curdiff = std::abs(feat[i] - htkfeat[i]);
    sum += curdiff;
    if (max < curdiff) {
      max = curdiff;
    }
  }
  std::cerr << "| Max diff across all dimensions " << max << "\n"; // 0.325853

  std::cerr << "| Avg diff across all dimensions " << sum / feat.size()
            << "\n"; // 0.00252719
}

TEST(MfccTest, BatchingTest) {
  int Tmax = 10000;
  auto input = randVec<float>(Tmax);
  FeatureParams featparams;
  featparams.deltaWindow = 0;
  featparams.frameSizeMs = 25;
  std::vector<bool> energies = {true, false};
  std::vector<bool> rawEnergies = {true, false};
  std::vector<bool> zMeans = {true, false};
  std::vector<bool> usePow = {true, false};

  int numTrials = 3;
  for (auto e : energies) {
    for (auto r : rawEnergies) {
      for (auto z : zMeans) {
        for (auto p : usePow) {
          featparams.useEnergy = e;
          featparams.rawEnergy = r;
          featparams.zeroMeanFrame = z;
          featparams.usePower = p;

          Mfcc mfcc(featparams);

          auto output = mfcc.apply(input);
          for (int i = 0; i < numTrials; ++i) {
            int chunkSz = 500 + (1000 * i) % 5000, curSz = 0;
            while (curSz + chunkSz < Tmax) {
              curSz += chunkSz;
              std::vector<float> curInput(curSz);
              std::copy(input.begin(), input.begin() + curSz, curInput.begin());
              auto curOutput = mfcc.apply(curInput);
              ASSERT_GT(curOutput.size(), 0);
              for (int j = 0; j < curOutput.size(); ++j) {
                ASSERT_NEAR(curOutput[j], output[j], 1E-4);
              }
            }
          }
        }
      }
    }
  }
}

TEST(MfccTest, BatchingTest2) {
  int Tmax = 10000;
  int batchSz = 100;
  auto input = randVec<float>(Tmax);
  FeatureParams featparams;
  featparams.frameSizeMs = 25;
  std::vector<bool> energies = {true, false};
  std::vector<bool> rawEnergies = {true, false};
  std::vector<bool> zMeans = {true, false};
  std::vector<bool> usePow = {true, false};

  for (auto e : energies) {
    for (auto r : rawEnergies) {
      for (auto z : zMeans) {
        for (auto p : usePow) {
          featparams.useEnergy = e;
          featparams.rawEnergy = r;
          featparams.zeroMeanFrame = z;
          featparams.usePower = p;

          Mfcc mfcc(featparams);

          auto output = mfcc.batchApply(input, batchSz);

          auto perBatchOutSz = output.size() / batchSz;
          auto perBatchInSz = input.size() / batchSz;
          for (int i = 0; i < batchSz; ++i) {
            std::vector<float> curInput(perBatchInSz);
            std::copy(
                input.begin() + i * perBatchInSz,
                input.begin() + (i + 1) * perBatchInSz,
                curInput.begin());
            auto curOutput = mfcc.apply(curInput);
            ASSERT_EQ(curOutput.size(), perBatchOutSz);
            for (int j = 0; j < curOutput.size(); ++j) {
              ASSERT_NEAR(curOutput[j], output[j + i * perBatchOutSz], 1E-4);
            }
          }
        }
      }
    }
  }
}

TEST(MfccTest, EmptyTest) {
  std::vector<float> input;
  FeatureParams featparams;
  Mfcc mfcc(featparams);
  auto output = mfcc.apply(input);
  ASSERT_TRUE(output.empty());

  int Tmax = 500;
  for (int t = 1; t <= Tmax; ++t) {
    input = randVec<float>(Tmax);
    output = mfcc.apply(input);
    ASSERT_TRUE(output.size() >= 0);
  }
}

TEST(MfccTest, ZeroInputTest) {
  auto params = FeatureParams();
  params.useEnergy = false;
  Mfsc mfcc(params);
  auto input = std::vector<float>(10000, 0.0);
  auto output = mfcc.apply(input);
  for (auto o : output) {
    ASSERT_NEAR(o, 0.0, 1E-4);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Resolve directory for data
#ifdef FEATURE_TEST_DATADIR
  loadPath = FEATURE_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}

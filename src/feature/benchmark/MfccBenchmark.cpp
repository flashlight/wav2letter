/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>

#include "libraries/feature/Mfcc.h"

using namespace w2l;

int main() {
  FeatureParams params;
  params.samplingFreq = 16000;
  params.frameSizeMs = 25;
  params.frameStrideMs = 10;
  params.numFilterbankChans = 20;
  params.lowFreqFilterbank = 0;
  params.highFreqFilterbank = 8000;
  params.numCepstralCoeffs = 13;
  params.lifterParam = 22;
  params.deltaWindow = 2;
  params.accWindow = 2;
  params.zeroMeanFrame = false;
  params.useEnergy = false;
  params.usePower = false;
  params.windowType = WindowType::HANNING;
  Mfcc<float> mfcc(params);

  std::vector<int> audiotimesec = {1, 10, 15, 20, 25, 50, 100}; // in seconds
  int64_t ntimes = 1000;

  std::cout << "Benchmark MFCC" << std::endl;
  for (int64_t t : audiotimesec) {
    double total_time = 0.0;
    std::vector<float> input(t * params.samplingFreq);
    for (int64_t M = 0; M < ntimes; M++) {
      std::generate(input.begin(), input.end(), []() {
        return (rand() * 1.0 / RAND_MAX);
      });
      auto start = std::chrono::system_clock::now();
      auto output = mfcc.apply(input);
      auto end = std::chrono::system_clock::now();
      total_time +=
          (std::chrono::duration_cast<std::chrono::milliseconds>(end - start))
              .count();
    }
    std::cout << "| Input Size : " << t << " sec , Avg. time taken "
              << std::setprecision(5) << total_time / ntimes << " msec"
              << std::endl;
  }
  return 0;
}

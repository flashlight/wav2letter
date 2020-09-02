/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <string>
#include <vector>

#include "Dataset.h"

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(infile, "", "Input path for pseudo-labeled lst file");
DEFINE_string(groundtruthfile, "", "Input path for ground truth lst file");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  auto predictionDict =
      filter::dataset::createTranscriptDictFromFile(FLAGS_infile);
  auto groundtruthDict =
      filter::dataset::createTranscriptDictFromFile(FLAGS_groundtruthfile);

  fl::EditDistanceMeter wer;
  size_t predictionDuration{0};
  for (auto& sample : predictionDict) {
    auto prediction = sample.second;
    auto groundtruth = groundtruthDict[sample.first];

    predictionDuration += prediction->getDuration();
    wer.add(prediction->transcriptWords, groundtruth->transcriptWords);
  }

  size_t groundtruthDuration{0};
  for (auto& sample : groundtruthDict) {
    groundtruthDuration += sample.second->getDuration();
  }

  // Num samples
  std::cout << "Prediction samples / groundtruth samples = "
            << predictionDict.size() << " / " << groundtruthDict.size() << " = "
            << (float)predictionDict.size() / (float)groundtruthDict.size()
            << std::endl;
  // Duration
  std::cout << "Prediction duration / groundtruth duration = "
            << predictionDuration << " / " << groundtruthDuration
            << " (seconds) = " << predictionDuration / (60.0 * 60.0 * 1000.0)
            << " / " << groundtruthDuration / (60.0 * 60.0 * 1000.0)
            << " (hours) = "
            << (float)predictionDuration / (float)groundtruthDuration
            << std::endl;
  // WER
  std::cout << "WER is " << wer.value()[0] << std::endl;

  return 0;
}

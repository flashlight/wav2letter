/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <flashlight/ext/common/Serializer.h>
#include <flashlight/lib/common/String.h>

#include "recipes/utilities/convlm_serializer/Utils.h"

using std::shared_ptr;
using std::string;
using std::vector;

int main(int argc, char** argv) {
  string exec(argv[0]);
  vector<string> argvs;
  gflags::SetUsageMessage(
      "Serializer of ConvLM from fairseq package. Usage: \n " + exec +
      std::string() + " [arc_path] [weight_path] [save_path]" + std::string() +
      " [outputTokensDim] [0-adaptiveSoftmax/1-CrossEntropy]" +
      " [0-loadCriterion/1-saveActivation] {10,20,30} {inputSize}");
  if (argc < 6) {
    FL_LOG(fl::FATAL) << gflags::ProgramUsage();
  }
  // parse params
  const char* archFile = argv[1];
  const char* weightFile = argv[2];
  const char* savePath = argv[3];
  int outputTokensDim = std::stoi(argv[4]);
  int criterionType = std::stoi(argv[5]);

  if (criterionType == 0 and argc != 9) {
    FL_LOG(fl::FATAL) << gflags::ProgramUsage()
                      << "\nFor Adaptive Softmax Criterion two parameters"
                      << " (tail and inputSize) should be also provided ";
  }
  vector<int> adaptiveTail;
  int inputSizeAdaptiveSoftmax = -1;
  bool loadActivation = false;

  // create network and criterion
  shared_ptr<fl::Module> network;
  shared_ptr<fl::BinaryModule> criterion = nullptr;

  if (criterionType == 0) {
    for (const auto& val : fl::lib::splitOnAnyOf(",", argv[7], true)) {
      adaptiveTail.push_back(std::stoi(val));
    }
    if (outputTokensDim > adaptiveTail.back()) {
      adaptiveTail.push_back(outputTokensDim);
    } else {
      if (outputTokensDim < adaptiveTail.back()) {
        FL_LOG(fl::FATAL)
            << "[ConvLMSerializer]: cannot specify adaptive softmax tail"
            << " larger than vocab size";
      }
    }
    inputSizeAdaptiveSoftmax = std::stoi(argv[8]);
    loadActivation = std::stoi(argv[6]);
  }

  FL_LOG(fl::INFO) << "[ConvLMSerializer]: Load convlm model";
  loadConvLM(
      network,
      criterion,
      archFile,
      weightFile,
      outputTokensDim,
      adaptiveTail,
      inputSizeAdaptiveSoftmax);

  if (loadActivation and
      std::dynamic_pointer_cast<fl::AdaptiveSoftMaxLoss>(criterion) !=
          nullptr) {
    auto as = std::dynamic_pointer_cast<fl::AdaptiveSoftMaxLoss>(criterion)
                  ->getActivation();
    std::dynamic_pointer_cast<fl::Sequential>(network)->add(as);
    criterion = nullptr;
  }

  network->eval();
  if (criterion != nullptr) {
    criterion->eval();
  }

  FL_LOG(fl::INFO) << "[ConvLMSerializer]: Saving into file " << savePath;
  fl::ext::Serializer::save(savePath, "2.0", network, criterion);

  return 0;
}

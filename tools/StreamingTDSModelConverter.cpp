/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
#include "inference/module/feature/feature.h"
#include "inference/module/module.h"
#include "inference/module/nn/nn.h"
#include "libraries/common/Dictionary.h"
#include "module/SpecAugment.h"
#include "module/TDSBlock.h"
#include "module/W2lModule.h"
#include "runtime/runtime.h"

DEFINE_string(outdir, "", "");

using namespace w2l;

namespace {
std::shared_ptr<streaming::ModuleParameter> variableToModuleParam(
    const fl::Variable& var) {
  LOG_IF(FATAL, var.type() != af::dtype::f32) << "Invalid variable type";
  std::vector<float> hostVec(var.elements());
  var.host(hostVec.data());
  return std::make_shared<streaming::ModuleParameter>(
      streaming::DataType::FLOAT, hostVec.data(), hostVec.size());
}

std::shared_ptr<streaming::LayerNorm>
convertLayerNorm(int featSz, const fl::Variable& wt, const fl::Variable& bs) {
  if (wt.elements() != 1 || bs.elements() != 1) {
    LOG(FATAL) << "Invalid params for layernorm.";
  }
  return std::make_shared<streaming::LayerNorm>(
      featSz, wt.scalar<float>(), bs.scalar<float>());
}

std::shared_ptr<streaming::Conv1d> convertConv1d(
    int cIn,
    int cOut,
    int kw,
    int dw,
    std::pair<int, int> padding,
    int groups,
    const fl::Variable& wt,
    const fl::Variable& bs) {
  if (wt.elements() != (cIn * cOut * kw) / (groups * groups) ||
      bs.elements() != cOut / groups) {
    LOG(INFO) << "Invalid params for conv1d. wt.elements():" << wt.elements()
              << " cIn:" << cIn << " cOut:" << cOut << " kw:" << kw
              << " groups:" << groups << "  bs.elements():" << bs.elements();
  }
  if (padding.first == -1 && padding.second == -1) {
    auto totalPad = kw - dw;
    padding.first = (totalPad + 1) / 2;
    padding.second = (totalPad + 1) / 2;
  } else if (padding.first == -1) {
    padding.first = kw - dw - padding.second;
  } else if (padding.second == -1) {
    padding.second = kw - dw - padding.first;
  }
  return streaming::createConv1d(
      cIn,
      cOut,
      kw,
      dw,
      padding,
      groups,
      variableToModuleParam(fl::reorder(wt, 2, 1, 0)),
      variableToModuleParam(bs));
}

std::shared_ptr<streaming::Linear> convertLinear(
    int nIn,
    int nOut,
    const fl::Variable& wt,
    const fl::Variable& bs) {
  if (wt.elements() != (nIn * nOut) || bs.elements() != nOut) {
    LOG(FATAL) << "Invalid params for linear.";
  }
  return streaming::createLinear(
      nIn, nOut, variableToModuleParam(wt), variableToModuleParam(bs));
}

std::shared_ptr<streaming::TDSBlock> convertTDS(
    int channels,
    int kernelSz,
    int featSz,
    int rightPad,
    std::vector<fl::Variable> params,
    int innerLinearDim) {
  auto conv1 = convertConv1d(
      channels * featSz,
      channels * featSz,
      kernelSz,
      1,
      {-1, rightPad},
      featSz,
      params[0],
      params[1]);
  if (innerLinearDim == 0) {
    innerLinearDim = featSz * channels;
  }
  auto lnorm1 = convertLayerNorm(featSz * channels, params[2], params[3]);
  auto lnorm2 = convertLayerNorm(featSz * channels, params[8], params[9]);
  auto lin1 =
      convertLinear(featSz * channels, innerLinearDim, params[4], params[5]);
  auto lin2 =
      convertLinear(innerLinearDim, featSz * channels, params[6], params[7]);
  return std::make_shared<streaming::TDSBlock>(
      conv1,
      lnorm1,
      lin1,
      lin2,
      lnorm2,
      streaming::DataType::FLOAT,
      streaming::DataType::FLOAT);
}

} // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  W2lSerializer::load(FLAGS_am, cfg, network, criterion);
  network->eval();
  criterion->eval();

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified " + dictPath);
  }
  Dictionary tokenDict(dictPath);
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  } else if (FLAGS_criterion != kAsgCriterion) {
    LOG(FATAL) << "This script currently support only CTC/ASG criterion";
  }
  int numTokens = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numTokens;

  int nFeat = 0;
  if (FLAGS_mfsc) {
    nFeat = FLAGS_filterbanks;
  } else {
    LOG(FATAL) << "This script currently support only mfsc features";
  }

  auto lines = w2l::getFileContent(pathsConcat(FLAGS_archdir, FLAGS_arch));

  auto streamingModule = std::make_shared<streaming::Sequential>();
  auto params = network->params();
  int curFeatSz = nFeat;
  int paramIdx = 0;
  int leftPad = -1, rightPad = -1;
  for (size_t i = 0; i < lines.size(); ++i) {
    auto columns = w2l::splitOnWhitespace(lines[i], true);
    if (columns.empty()) {
      continue;
    }
    auto layerType = columns[0];
    if (layerType == "C2") {
      if (columns.size() < 8) {
        LOG(FATAL) << "Invalid arch specified for C2";
      }
      auto conv1d = convertConv1d(
          std::stoi(columns[1]) * nFeat,
          std::stoi(columns[2]) * nFeat,
          std::stoi(columns[3]),
          std::stoi(columns[5]),
          {leftPad, rightPad},
          nFeat,
          params[paramIdx],
          params[paramIdx + 1]);
      streamingModule->add(conv1d);
      leftPad = -1;
      rightPad = -1;
      paramIdx += 2;
      curFeatSz = std::stoi(columns[2]) * nFeat;
    } else if (layerType == "PD") {
      if (columns.size() != 4) {
        LOG(FATAL) << "Padding is supported only along time axis";
      }
      if (!startsWith(lines[i + 1], "C2")) {
        LOG(FATAL) << "Padding layer must be followed by conv layer";
      }
      leftPad = std::stoi(columns[2]);
      rightPad = std::stoi(columns[3]);
    } else if (layerType == "R") {
      streamingModule->add(
          std::make_shared<streaming::Relu>(streaming::DataType::FLOAT));
    } else if (layerType == "LN") {
      if (columns[1] != "1" || columns[2] != "2") {
        LOG(FATAL)
            << "Unsupported LayerNorm axis: must be {1, 2} for streaming";
      }
      auto lyrNorm =
          convertLayerNorm(curFeatSz, params[paramIdx], params[paramIdx + 1]);
      streamingModule->add(lyrNorm);
      paramIdx += 2;
    } else if (layerType == "L") {
      int outDim = (columns[2] == "NLABEL") ? numTokens : std::stoi(columns[2]);
      int inDim = std::stoi(columns[1]);
      if (params.size() < paramIdx + 2) {
        LOG(FATAL) << "Error serializing Linear module. Not enough parameters.";
      }
      auto linear =
          convertLinear(inDim, outDim, params[paramIdx], params[paramIdx + 1]);
      streamingModule->add(linear);
      paramIdx += 2;
    } else if (layerType == "TDS") {
      auto stds = convertTDS(
          std::stoi(columns[1]),
          std::stoi(columns[2]),
          std::stoi(columns[3]),
          (columns.size() > 6) ? std::stoi(columns[6]) : -1,
          {params.begin() + paramIdx, params.begin() + paramIdx + 10},
          (columns.size() > 5) ? std::stoi(columns[5]) : 0);
      streamingModule->add(stds);
      paramIdx += 10;
    } else if (layerType == "V") {
      std::cerr << "Skipping View module: " << lines[i] << std::endl;
    } else if (layerType == "RO") {
      std::cerr << "Skipping Reorder module: " << lines[i] << std::endl;
    } else if (layerType == "DO") {
      std::cerr << "Skipping Dropout module: " << lines[i] << std::endl;
    } else if (layerType == "SAUG") {
      std::cerr << "Skipping SpecAugment module: " << lines[i] << std::endl;
    } else if (layerType != "#") {
      throw std::logic_error("Unrecognized/unparsable line " + lines[i]);
    }
  }

  {
    std::string amFilePath = pathsConcat(FLAGS_outdir, "acoustic_model.bin");
    std::ofstream amFile(amFilePath, std::ios::binary);
    LOG(INFO) << "Serializing acoustic model to '" << amFilePath << "'";

    if (!amFile.is_open()) {
      throw std::runtime_error("failed to open file for reading");
    }
    cereal::BinaryOutputArchive ar(amFile);
    ar(streamingModule);
  }

  {
    std::string tokenFilePath = pathsConcat(FLAGS_outdir, "tokens.txt");
    std::ofstream tokenFile(tokenFilePath);
    LOG(INFO) << "Writing tokens file to '" << tokenFilePath << "'";
    for (int i = 0; i < tokenDict.indexSize(); ++i) {
      tokenFile << tokenDict.getEntry(i) << "\n";
    }
    tokenFile.close();
  }

  if (FLAGS_criterion == kAsgCriterion) {
    if (criterion->params().size() == 0 ||
        criterion->param(0).elements() !=
            tokenDict.indexSize() * tokenDict.indexSize()) {
      throw std::runtime_error("Invalid criterion parameters for ASG");
    }
    std::string transitionsFilePath =
        pathsConcat(FLAGS_outdir, "transitions.bin");
    std::ofstream transitionsFile(transitionsFilePath);
    LOG(INFO) << "Writing transitions file to '" << transitionsFilePath << "'";
    std::vector<float> transitionsVec(criterion->param(0).elements());
    criterion->param(0).host(transitionsVec.data());
    fl::save(transitionsFile, transitionsVec);
    transitionsFile.close();
  }

  {
    std::string featFilePath =
        pathsConcat(FLAGS_outdir, "feature_extractor.bin");
    std::ofstream featFile(featFilePath, std::ios::binary);
    LOG(INFO) << "Serializing feature extraction model to '" << featFilePath
              << "'";
    auto featureModule = std::make_shared<streaming::Sequential>();
    featureModule->add(std::make_shared<streaming::LogMelFeature>(nFeat));
    LOG_IF(FATAL, FLAGS_localnrmlleftctx <= 0)
        << "Local Norm should be used for online inference";
    featureModule->add(std::make_shared<streaming::LocalNorm>(
        nFeat, FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx));

    if (!featFile.is_open()) {
      throw std::runtime_error("failed to open file for reading");
    }
    cereal::BinaryOutputArchive ar(featFile);
    ar(featureModule);
  }

  LOG(INFO) << "verifying serialization ...";
  af::array inputArr = af::randu(16, 80);
  auto outputArr = network->forward({fl::Variable(inputArr, false)})[0].array();
  std::vector<float> outputVec(outputArr.elements());
  outputArr.host(outputVec.data());

  std::vector<float> inputVec(inputArr.elements());
  af::reorder(inputArr, 2, 1, 0).host(inputVec.data());
  auto inputState = std::make_shared<streaming::ModuleProcessingState>(1);
  std::shared_ptr<streaming::IOBuffer> inputBuffer = inputState->buffer(0);
  inputBuffer->write<float>(inputVec.data(), inputVec.size());

  streamingModule->start(inputState);
  auto outputState = streamingModule->run(inputState);
  streamingModule->finish(inputState);

  std::shared_ptr<streaming::IOBuffer> outputBuffer = outputState->buffer(0);

  LOG_IF(FATAL, outputBuffer->size<float>() != outputVec.size())
      << "[Serialization Error] Incorrect output sizes";
  float* outPtr = outputBuffer->data<float>();
  for (int i = 0; i < outputBuffer->size<float>(); i++) {
    float streamingOut = outPtr[i];
    float w2lOut = outputVec[i];
    LOG_IF(FATAL, fabs(streamingOut - w2lOut) > 1e-2)
        << "[Serialization Error] Mismatched output w2l:" << w2lOut
        << " vs streaming:" << streamingOut;
  }
  LOG(INFO) << "Done !";
  return 0;
}

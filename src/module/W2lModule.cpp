/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "W2lModule.h"

#include <string>

#include <glog/logging.h>

#include "common/Utils.h"

using namespace fl;

namespace {
std::shared_ptr<Module> parseLine(const std::string& line);
}

namespace w2l {

std::shared_ptr<Sequential> createW2lSeqModule(
    const std::string& archfile,
    int64_t nFeatures,
    int64_t nClasses) {
  auto net = std::make_shared<Sequential>();
  auto layers = getFileContent(archfile);

  for (auto& l : layers) {
    std::string lrepl = trim(l);
    replaceAll(lrepl, "NFEAT", std::to_string(nFeatures));
    replaceAll(lrepl, "NLABEL", std::to_string(nClasses));

    if (lrepl.empty() || startsWith(lrepl, "#")) {
      continue; // ignore empty lines / comments
    } else {
      net->add(parseLine(lrepl));
    }
  }
  return net;
}

} // namespace w2l

namespace {
std::shared_ptr<Module> parseLine(const std::string& line) {
  auto params = w2l::splitOnWhitespace(line, true);

  auto inRange = [&](const int a, const int b, const int c) {
    return (a <= b && b <= c);
  };

  /* ========== TRANSFORMATIONS ========== */

  if ((params[0] == "RO") || (params[0] == "V")) {
    LOG_IF(FATAL, params.size() != 5) << "Failed parsing - " << line;
    int dim1 = std::stoi(params[1]);
    int dim2 = std::stoi(params[2]);
    int dim3 = std::stoi(params[3]);
    int dim4 = std::stoi(params[4]);
    if (params[0] == "RO") {
      return std::make_shared<Reorder>(dim1, dim2, dim3, dim4);
    } else {
      return std::make_shared<View>(af::dim4(dim1, dim2, dim3, dim4));
    }
  }

  if (params[0] == "PD") {
    LOG_IF(FATAL, !inRange(4, params.size(), 10) || (params.size() & 1))
        << "Failed parsing - " << line;
    auto val = std::stod(params[1]);
    params.resize(10, "0");
    std::pair<int, int> pad0 = {std::stoi(params[2]), std::stoi(params[3])};
    std::pair<int, int> pad1 = {std::stoi(params[4]), std::stoi(params[5])};
    std::pair<int, int> pad2 = {std::stoi(params[6]), std::stoi(params[7])};
    std::pair<int, int> pad3 = {std::stoi(params[8]), std::stoi(params[9])};
    return std::make_shared<Padding>(pad0, pad1, pad2, pad3, val);
  }

  /* ========== CONVOLUTIONS ========== */

  if (params[0] == "C" || params[0] == "C1") {
    LOG_IF(FATAL, !inRange(5, params.size(), 6)) << "Failed parsing - " << line;
    int cisz = std::stoi(params[1]);
    int cosz = std::stoi(params[2]);
    int cwx = std::stoi(params[3]);
    int csx = std::stoi(params[4]);
    int cpx = (params.size() == 6) ? std::stoi(params[5]) : 0;
    return std::make_shared<Conv2D>(cisz, cosz, cwx, 1, csx, 1, cpx, 0);
  }

  if (params[0] == "C2") {
    LOG_IF(FATAL, !inRange(7, params.size(), 9)) << "Failed parsing - " << line;
    int cisz = std::stoi(params[1]);
    int cosz = std::stoi(params[2]);
    int cwx = std::stoi(params[3]);
    int cwy = std::stoi(params[4]);
    int csx = std::stoi(params[5]);
    int csy = std::stoi(params[6]);
    int cpx = (params.size() >= 8) ? std::stoi(params[7]) : 0;
    int cpy = (params.size() >= 9) ? std::stoi(params[8]) : 0;
    return std::make_shared<Conv2D>(cisz, cosz, cwx, cwy, csx, csy, cpx, cpy);
  }

  /* ========== LINEAR ========== */

  if (params[0] == "L") {
    LOG_IF(FATAL, params.size() != 3) << "Failed parsing - " << line;
    int lisz = std::stoi(params[1]);
    int losz = std::stoi(params[2]);
    return std::make_shared<Linear>(lisz, losz);
  }

  /* ========== NORMALIZATIONS ========== */

  if (params[0] == "BN") {
    LOG_IF(FATAL, !inRange(3, params.size(), 5)) << "Failed parsing - " << line;
    int featSz = std::stoi(params[1]);
    std::vector<int> featDims;
    for (int i = 2; i < params.size(); ++i) {
      featDims.emplace_back(std::stoi(params[i]));
    }
    return std::make_shared<BatchNorm>(featDims, featSz);
  }

  if (params[0] == "LN") {
    LOG_IF(FATAL, !inRange(2, params.size(), 4)) << "Failed parsing - " << line;
    std::vector<int> featDims;
    for (int i = 1; i < params.size(); ++i) {
      featDims.emplace_back(std::stoi(params[i]));
    }
    return std::make_shared<LayerNorm>(featDims);
  }

  if (params[0] == "WN") {
    LOG_IF(FATAL, params.size() < 3) << "Failed parsing - " << line;
    int dim = std::stoi(params[1]);
    std::string childStr = w2l::join(" ", params.begin() + 2, params.end());
    return std::make_shared<WeightNorm>(parseLine(childStr), dim);
  }

  if (params[0] == "DO") {
    LOG_IF(FATAL, params.size() != 2) << "Failed parsing - " << line;
    auto drpVal = std::stod(params[1]);
    return std::make_shared<Dropout>(drpVal);
  }

  /* ========== POOLING ========== */

  if ((params[0] == "M") || (params[0] == "A")) {
    LOG_IF(FATAL, params.size() < 5) << "Failed parsing - " << line;
    int wx = std::stoi(params[1]);
    int wy = std::stoi(params[2]);
    int dx = std::stoi(params[3]);
    int dy = std::stoi(params[4]);
    int px = params.size() > 5 ? std::stoi(params[5]) : 0;
    int py = params.size() > 6 ? std::stoi(params[6]) : 0;
    auto mode = (params[0] == "A") ? PoolingMode::AVG_INCLUDE_PADDING
                                   : PoolingMode::MAX;

    return std::make_shared<Pool2D>(wx, wy, dx, dy, px, py, mode);
  }

  /* ========== ACTIVATIONS ========== */

  if (params[0] == "ELU") {
    return std::make_shared<ELU>();
  }

  if (params[0] == "R") {
    return std::make_shared<ReLU>();
  }

  if (params[0] == "PR") {
    auto numParams = params.size() > 1 ? std::stoi(params[1]) : 1;
    auto initVal = params.size() > 2 ? std::stod(params[2]) : 0.25;
    return std::make_shared<PReLU>(numParams, initVal);
  }

  if (params[0] == "LG") {
    return std::make_shared<Log>();
  }

  if (params[0] == "HT") {
    return std::make_shared<HardTanh>();
  }

  if (params[0] == "T") {
    return std::make_shared<Tanh>();
  }

  if (params[0] == "GLU") {
    LOG_IF(FATAL, params.size() != 2) << "Failed parsing - " << line;
    int dim = std::stoi(params[1]);
    return std::make_shared<GatedLinearUnit>(dim);
  }

  if (params[0] == "LSM") {
    LOG_IF(FATAL, params.size() != 2) << "Failed parsing - " << line;
    int dim = std::stoi(params[1]);
    return std::make_shared<LogSoftmax>(dim);
  }

  /* ========== RNNs ========== */

  auto rnnLayer = [&](const std::vector<std::string>& prms, RnnMode mode) {
    int iSz = std::stoi(prms[1]);
    int oSz = std::stoi(prms[2]);
    int numLayers = (prms.size() > 3) ? std::stoi(prms[3]) : 1;
    bool bidirectional = (prms.size() > 4) ? std::stoi(prms[4]) > 0 : false;
    float dropout = (prms.size() > 5) ? std::stof(prms[5]) : 0.0;
    return std::make_shared<RNN>(
        iSz, oSz, numLayers, mode, bidirectional, dropout);
  };

  if (params[0] == "RNN") {
    LOG_IF(FATAL, params.size() < 3) << "Failed parsing - " << line;
    return rnnLayer(params, RnnMode::RELU);
  }

  if (params[0] == "GRU") {
    LOG_IF(FATAL, params.size() < 3) << "Failed parsing - " << line;
    return rnnLayer(params, RnnMode::GRU);
  }

  if (params[0] == "LSTM") {
    LOG_IF(FATAL, params.size() < 3) << "Failed parsing - " << line;
    return rnnLayer(params, RnnMode::LSTM);
  }

  LOG(FATAL) << "Failed parsing - " << line;
  return nullptr;
}
} // namespace

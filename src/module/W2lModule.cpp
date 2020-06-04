/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include <glog/logging.h>

#include "W2lModule.h"

#include "common/FlashlightUtils.h"
#include "module/SpecAugment.h"
#include "module/TDSBlock.h"

#ifdef W2L_BUILD_FB_DEPENDENCIES
#include "experimental/frontend/Frontend.h"
#endif

using namespace fl;

namespace {
std::shared_ptr<Module> parseLine(const std::string& line);

std::shared_ptr<Module> parseLines(
    const std::vector<std::string>& lines,
    const int lineIdx,
    int& numLinesParsed);
} // namespace

namespace w2l {

std::shared_ptr<Sequential> createW2lSeqModule(
    const std::string& archfile,
    int64_t nFeatures,
    int64_t nClasses) {
  auto net = std::make_shared<Sequential>();
  auto layers = getFileContent(archfile);
  int numLinesParsed = 0;

  // preprocess
  std::vector<std::string> processedLayers;
  for (auto& l : layers) {
    std::string lrepl = trim(l);
    replaceAll(lrepl, "NFEAT", std::to_string(nFeatures));
    replaceAll(lrepl, "NLABEL", std::to_string(nClasses));

    if (lrepl.empty() || startsWith(lrepl, "#")) {
      continue; // ignore empty lines / comments
    }
    processedLayers.emplace_back(lrepl);
  }

  int lid = 0;
  while (lid < processedLayers.size()) {
    net->add(parseLines(processedLayers, lid, numLinesParsed));
    lid += (numLinesParsed + 1);
  }

  return net;
}

} // namespace w2l

namespace {
std::shared_ptr<Module> parseLine(const std::string& line) {
  int dummy;
  return parseLines({line}, 0, dummy);
}

std::shared_ptr<Module> parseLines(
    const std::vector<std::string>& lines,
    const int lineIdx,
    int& numLinesParsed) {
  auto line = lines[lineIdx];
  numLinesParsed = 0;
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

  /* ========== TRANSFORMERS ========== */

  if (params[0] == "TR") {
    int modelDim = std::stoi(params[1]);
    int mlpDim = std::stoi(params[2]);
    int nHead = std::stoi(params[3]);
    int csz = std::stoi(params[4]);
    float pDropout = std::stof(params[5]);
    float pLayerdrop = (params.size() >= 7) ? std::stof(params[6]) : 0.0;
    int preLN = (params.size() >= 8) ? std::stoi(params[7]) : 0;
    return std::make_shared<Transformer>(
        modelDim,
        modelDim / nHead,
        mlpDim,
        nHead,
        csz,
        pDropout,
        pLayerdrop,
        false,
        preLN);
  }

  if (params[0] == "POSEMB") {
    int layerDim = std::stoi(params[1]);
    int csz = std::stoi(params[2]);
    float dropout = (params.size() >= 4) ? std::stof(params[3]) : 0.0;
    return std::make_shared<PositionEmbedding>(layerDim, csz, dropout);
  }

  /* ========== CONVOLUTIONS ========== */

  if (params[0] == "C" || params[0] == "C1") {
    LOG_IF(FATAL, !inRange(5, params.size(), 7)) << "Failed parsing - " << line;
    int cisz = std::stoi(params[1]);
    int cosz = std::stoi(params[2]);
    int cwx = std::stoi(params[3]);
    int csx = std::stoi(params[4]);
    int cpx = (params.size() >= 6) ? std::stoi(params[5]) : 0;
    int cdx = (params.size() >= 7) ? std::stoi(params[6]) : 1;
    return std::make_shared<Conv2D>(cisz, cosz, cwx, 1, csx, 1, cpx, 0, cdx, 1);
  }

  if (params[0] == "TDS") {
    LOG_IF(FATAL, !inRange(4, params.size(), 8)) << "Failed parsing - " << line;
    int cisz = std::stoi(params[1]);
    int cwx = std::stoi(params[2]);
    int freqdim = std::stoi(params[3]);
    double dropprob = (params.size() >= 5 ? std::stod(params[4]) : 0);
    int l2 = (params.size() >= 6 ? std::stoi(params[5]) : 0);
    int rPad = (params.size() >= 7) ? std::stoi(params[6]) : -1;
    bool lNormIncludeTime =
        (params.size() >= 8 && std::stoi(params[7]) == 0) ? false : true;
    return std::make_shared<w2l::TDSBlock>(
        cisz, cwx, freqdim, dropprob, l2, rPad, lNormIncludeTime);
  }

  if (params[0] == "AC") {
    LOG_IF(FATAL, !inRange(5, params.size(), 8)) << "Failed parsing - " << line;
    int cisz = std::stoi(params[1]);
    int cosz = std::stoi(params[2]);
    int cwx = std::stoi(params[3]);
    int csx = std::stoi(params[4]);
    int cpx = (params.size() >= 6) ? std::stoi(params[5]) : 0;
    float futurePartPx = (params.size() >= 7) ? std::stof(params[6]) : 1.;
    int cdx = (params.size() >= 8) ? std::stoi(params[7]) : 1;
    return std::make_shared<AsymmetricConv1D>(
        cisz, cosz, cwx, csx, cpx, futurePartPx, cdx);
  }

  if (params[0] == "C2") {
    LOG_IF(FATAL, !inRange(7, params.size(), 11))
        << "Failed parsing - " << line;
    int cisz = std::stoi(params[1]);
    int cosz = std::stoi(params[2]);
    int cwx = std::stoi(params[3]);
    int cwy = std::stoi(params[4]);
    int csx = std::stoi(params[5]);
    int csy = std::stoi(params[6]);
    int cpx = (params.size() >= 8) ? std::stoi(params[7]) : 0;
    int cpy = (params.size() >= 9) ? std::stoi(params[8]) : 0;
    int cdx = (params.size() >= 10) ? std::stoi(params[9]) : 1;
    int cdy = (params.size() >= 11) ? std::stoi(params[10]) : 1;
    return std::make_shared<Conv2D>(
        cisz, cosz, cwx, cwy, csx, csy, cpx, cpy, cdx, cdy);
  }

  /* ========== LINEAR ========== */

  if (params[0] == "L") {
    LOG_IF(FATAL, !inRange(3, params.size(), 4)) << "Failed parsing - " << line;
    int lisz = std::stoi(params[1]);
    int losz = std::stoi(params[2]);
    bool bias = (params.size() == 4) && params[3] == "0" ? false : true;
    return std::make_shared<Linear>(lisz, losz, bias);
  }

  /* ========== EMBEDDING ========== */

  if (params[0] == "E") {
    LOG_IF(FATAL, params.size() != 3) << "Failed parsing - " << line;
    int embsz = std::stoi(params[1]);
    int ntokens = std::stoi(params[2]);
    return std::make_shared<Embedding>(embsz, ntokens);
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
    if (featDims == std::vector<int>{3}) {
      LOG(FATAL)
          << "flashlight LayerNorm API for specifying `featAxes` is modified "
          << "recently - https://git.io/Je70U. You probably would want to "
          << "specify LN 0 1 2 instead of LN 3. If you really know what you're "
          << "doing, comment out this check and build again.";
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

  if (params[0] == "R6") {
    return std::make_shared<ReLU6>();
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

  if (params[0] == "SH") {
    auto beta = params.size() > 1 ? std::stof(params[1]) : 1.0;
    return std::make_shared<Swish>(beta);
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

  /* ========== Residual block ========== */
  if (params[0] == "RES") {
    LOG_IF(FATAL, params.size() <= 3) << "Failed parsing - " << line;

    auto residualBlock = [&](const std::vector<std::string>& prms,
                             int& numResLayerAndSkip) {
      int numResLayers = std::stoi(prms[1]);
      int numSkipConnections = std::stoi(prms[2]);
      std::shared_ptr<Residual> resPtr = std::make_shared<Residual>();

      int numProjections = 0;

      for (int i = 1; i <= numResLayers + numSkipConnections; ++i) {
        LOG_IF(FATAL, lineIdx + i + numProjections >= lines.size())
            << "Failed parsing Residual block";
        std::string resLine = lines[lineIdx + i + numProjections];
        auto resLinePrms = w2l::splitOnWhitespace(resLine, true);

        if (resLinePrms[0] == "SKIP") {
          LOG_IF(FATAL, !inRange(3, resLinePrms.size(), 4))
              << "Failed parsing - " << resLine;
          resPtr->addShortcut(
              std::stoi(resLinePrms[1]), std::stoi(resLinePrms[2]));
          if (resLinePrms.size() == 4) {
            resPtr->addScale(
                std::stoi(resLinePrms[2]), std::stof(resLinePrms[3]));
          }
        } else if (resLinePrms[0] == "SKIPL") {
          LOG_IF(FATAL, !inRange(4, resLinePrms.size(), 5))
              << "Failed parsing - " << resLine;
          int numProjectionLayers = std::stoi(resLinePrms[3]);
          auto projection = std::make_shared<Sequential>();

          for (int j = 1; j <= numProjectionLayers; ++j) {
            LOG_IF(FATAL, lineIdx + i + numProjections + j >= lines.size())
                << "Failed parsing Projection block";
            projection->add(parseLine(lines[lineIdx + i + numProjections + j]));
          }
          resPtr->addShortcut(
              std::stoi(resLinePrms[1]), std::stoi(resLinePrms[2]), projection);
          if (resLinePrms.size() == 5) {
            resPtr->addScale(
                std::stoi(resLinePrms[2]), std::stof(resLinePrms[4]));
          }
          numProjections += numProjectionLayers;
        } else {
          resPtr->add(parseLine(resLine));
        }
      }

      numResLayerAndSkip = numResLayers + numSkipConnections + numProjections;
      return resPtr;
    };

    auto numBlocks = params.size() == 4 ? std::stoi(params.back()) : 1;
    LOG_IF(FATAL, numBlocks <= 0)
        << "Invalid number of residual blocks: " << numBlocks;

    if (numBlocks > 1) {
      auto res = std::make_shared<Sequential>();
      for (int n = 0; n < numBlocks; ++n) {
        res->add(residualBlock(params, numLinesParsed));
      }
      return res;
    } else {
      return residualBlock(params, numLinesParsed);
    }
  }

  /* ========== Data Augmentation  ========== */
  if (params[0] == "SAUG") {
    LOG_IF(FATAL, params.size() != 7) << "Failed parsing - " << line;
    return std::make_shared<w2l::SpecAugment>(
        std::stoi(params[1]),
        std::stoi(params[2]),
        std::stoi(params[3]),
        std::stoi(params[4]),
        std::stod(params[5]),
        std::stoi(params[6]));
  }

#ifdef W2L_BUILD_FB_DEPENDENCIES

  /* ========== Trainable frontend ========== */
  if (params[0] == "SL2P") {
    return std::make_shared<w2l::SqL2Pooling>();
  }

  if (params[0] == "LC") {
    LOG_IF(FATAL, params.size() < 2) << "Failed parsing - " << line;
    double k = std::stod(params[1]);
    return std::make_shared<w2l::LogCompression>(k);
  }

  if (params[0] == "LPF") {
    LOG_IF(FATAL, params.size() < 5) << "Failed parsing - " << line;

    int nin = std::stoi(params[1]);
    int kw = std::stoi(params[2]);
    int dw = std::stoi(params[3]);
    w2l::LPFMode learn = w2l::LPFMode::FIXED;
    if (params[4] == "L") {
      learn = w2l::LPFMode::LEARN;
    }
    return std::make_shared<w2l::Lowpass>(nin, kw, dw, learn);
  }

  if (params[0] == "WF") {
    LOG_IF(FATAL, params.size() < 3) << "Failed parsing - " << line;

    std::string weightsFile = params[1];
    std::string childStr = w2l::join(" ", params.begin() + 2, params.end());

    auto conv = parseLine(childStr);
    auto weights = conv->param(0);
    w2l::initializeWeights(weightsFile, weights);
    conv->setParams(weights, 0);
    return conv;
  }
#endif

  LOG(FATAL) << "Failed parsing - " << line;
  return nullptr;
}
} // namespace

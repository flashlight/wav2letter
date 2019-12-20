/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Featurize.h"

#include <math.h>
#include <fstream>
#include <vector>

#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "libraries/feature/Mfcc.h"
#include "libraries/feature/Mfsc.h"
#include "libraries/feature/PowerSpectrum.h"

namespace w2l {

namespace {

Mfcc& getMfcc() {
  static Mfcc mfcc(defineSpeechFeatureParams());
  return mfcc;
}

Mfsc& getMfsc() {
  static Mfsc mfsc(defineSpeechFeatureParams());
  return mfsc;
}

PowerSpectrum& getPowerSpectrum() {
  static PowerSpectrum powspec(defineSpeechFeatureParams());
  return powspec;
}

} // namespace

W2lFeatureData featurize(
    const std::vector<W2lLoaderData>& data,
    const DictionaryMap& dicts) {
  if (data.empty()) {
    return {};
  }
  auto batchSz = data.size();
  W2lFeatureData feat;
  std::vector<std::string> sampleIds;

  // Featurize Input
  size_t maxInSize = 0;
  for (const auto& d : data) {
    maxInSize = std::max(maxInSize, d.input.size());
  }
  int64_t T = maxInSize / FLAGS_channels;

  // CHANNELS X T X BATCHSZ (Col Major)
  std::vector<float> mergedInput(maxInSize * batchSz, 0.0);
  // Batch into a single array
  for (size_t b = 0; b < batchSz; ++b) {
    if (data[b].input.empty()) {
      continue;
    }
    std::copy(
        data[b].input.begin(),
        data[b].input.end(),
        mergedInput.begin() + b * maxInSize);
  }
  // T X CHANNELS X BATCHSZ (Col Major)
  auto inFeat =
      transpose2d<float>(std::move(mergedInput), T, FLAGS_channels, batchSz);
  feat.inputDims = af::dim4(T, FLAGS_channels, 1, batchSz);
  if (FLAGS_pow || FLAGS_mfsc || FLAGS_mfcc) {
    if ((FLAGS_mfcc && FLAGS_mfsc) || (FLAGS_pow && FLAGS_mfsc) ||
        (FLAGS_mfcc && FLAGS_pow)) {
      LOG(FATAL) << "Only one of -mfsc, -mfcc, -pow options can set to true";
    }
    int64_t featSz = 1;
    if (FLAGS_mfcc) {
      auto& mfcc = getMfcc();
      featSz = mfcc.getFeatureParams().mfccFeatSz();
      inFeat = mfcc.batchApply(inFeat, FLAGS_channels * batchSz);
    }
    if (FLAGS_mfsc) {
      auto& mfsc = getMfsc();
      featSz = mfsc.getFeatureParams().mfscFeatSz();
      inFeat = mfsc.batchApply(inFeat, FLAGS_channels * batchSz);
    }
    if (FLAGS_pow) {
      auto& powspec = getPowerSpectrum();
      featSz = powspec.getFeatureParams().powSpecFeatSz();
      inFeat = powspec.batchApply(inFeat, FLAGS_channels * batchSz);
    }
    T = inFeat.size() / (FLAGS_channels * batchSz * featSz);
    // Before: FEAT X FRAMES X CHANNELS X BATCHSIZE (Col Major)
    inFeat = transpose2d<float>(inFeat, T, featSz, FLAGS_channels * batchSz);
    // After: FRAMES X FEAT X CHANNELS X BATCHSIZE (Col Major)
    feat.inputDims = af::dim4(T, featSz, FLAGS_channels, batchSz);
  }

  if (FLAGS_localnrmlleftctx > 0 || FLAGS_localnrmlrightctx > 0) {
    feat.input = localNormalize(
        inFeat, FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx, T, batchSz);
  } else {
    feat.input = normalize(inFeat, batchSz);
  }

  // Featurize Target
  for (auto targetIter : data[0].targets) {
    auto targetType = targetIter.first;
    std::vector<std::vector<int>> tgtFeat;
    size_t maxTgtSize = 0;
    if (dicts.find(targetType) == dicts.end()) {
      LOG(FATAL) << "Dictionary not provided for target: " << targetType;
    }
    auto dict = dicts.find(targetType)->second;

    for (const auto& d : data) {
      if (d.targets.find(targetType) == d.targets.end()) {
        LOG(FATAL) << "Target type not found for featurization: " << targetType;
      }
      auto target = d.targets.find(targetType)->second;

      if (targetType == kTargetIdx) {
        auto tgtVec = dict.mapEntriesToIndices(target);
        if (!FLAGS_surround.empty()) {
          auto idx = dict.getIndex(FLAGS_surround);
          tgtVec.emplace_back(idx);
          if (tgtVec.size() > 1) {
            tgtVec.emplace_back(idx);
            std::rotate(tgtVec.begin(), tgtVec.end() - 1, tgtVec.end());
          }
        }
        if (FLAGS_replabel > 0) {
          tgtVec = packReplabels(tgtVec, dict, FLAGS_replabel);
        }
        if (FLAGS_criterion == kAsgCriterion) {
          uniq(tgtVec);
        }
        if (FLAGS_eostoken) {
          tgtVec.emplace_back(dict.getIndex(kEosToken));
        }
        tgtFeat.emplace_back(tgtVec);
        maxTgtSize = std::max(maxTgtSize, tgtVec.size());

        int padVal =
            FLAGS_eostoken ? dict.getIndex(kEosToken) : kTargetPadValue;
        // L X BATCHSZ (Col Major)
        feat.targets[targetType].resize(batchSz * maxTgtSize, padVal);
        feat.targetDims[targetType] = af::dim4(maxTgtSize, batchSz);
      } else if (targetType == kWordIdx) {
        auto tgtVec = dict.mapEntriesToIndices(target);
        tgtFeat.emplace_back(tgtVec);
        maxTgtSize = std::max(maxTgtSize, tgtVec.size());

        int padVal = dict.getIndex(kUnkToken);
        // L X BATCHSZ (Col Major)
        feat.targets[targetType].resize(batchSz * maxTgtSize, padVal);
        feat.targetDims[targetType] = af::dim4(maxTgtSize, batchSz);
      } else {
        LOG(FATAL) << "Unrecognized target type" << targetType;
      }
    }

    // Batch into a single array
    for (size_t i = 0; i < batchSz; ++i) {
      if (tgtFeat[i].empty()) {
        continue;
      }
      std::copy(
          tgtFeat[i].begin(),
          tgtFeat[i].end(),
          feat.targets[targetType].begin() + maxTgtSize * i);
    }
  }

  // Featurize sampleid
  size_t maxSampleIdLen = 0;
  for (size_t b = 0; b < batchSz; ++b) {
    sampleIds.emplace_back(data[b].sampleId);
    maxSampleIdLen = std::max(maxSampleIdLen, data[b].sampleId.size());
  }

  // Pack the sample ids
  // batchsize X maxSampleIdLen
  feat.sampleIds.resize(batchSz * maxSampleIdLen, -1);
  auto offset = 0;
  for (auto& sampleId : sampleIds) {
    auto charId = 0;
    for (auto& c : sampleId) {
      feat.sampleIds[offset + charId] = int(c);
      charId += 1;
    }
    std::transform(
        feat.sampleIds.begin() + offset,
        feat.sampleIds.begin() + offset + sampleId.size(),
        sampleId.begin(),
        [](unsigned char c) -> int { return int(c); });
    offset += maxSampleIdLen;
  }
  feat.sampleIdsDims = af::dim4(maxSampleIdLen, batchSz);

  return feat;
}

FeatureParams defineSpeechFeatureParams() {
  FeatureParams params;

  // PowerSpectrum, Mfsc, Mfcc
  params.samplingFreq = FLAGS_samplerate;
  params.frameSizeMs = FLAGS_framesizems;
  params.frameStrideMs = FLAGS_framestridems;
  params.lowFreqFilterbank = 0;
  params.highFreqFilterbank = FLAGS_samplerate / 2;
  params.zeroMeanFrame = false;
  params.ditherVal = 0.0;

  // Mfsc, Mfcc
  params.numFilterbankChans = FLAGS_filterbanks;
  params.useEnergy = false;
  params.usePower = false;
  params.accWindow = FLAGS_devwin;
  params.deltaWindow = FLAGS_devwin;

  //  Mfcc
  params.numCepstralCoeffs = FLAGS_mfcccoeffs;
  params.lifterParam = kLifterParam;
  params.melFloor = FLAGS_melfloor;

  return params;
}

int64_t getSpeechFeatureSize() {
  int64_t numFeatures = FLAGS_channels;
  auto featparams = defineSpeechFeatureParams();
  if (FLAGS_pow) {
    numFeatures = featparams.powSpecFeatSz();
  } else if (FLAGS_mfsc) {
    numFeatures = featparams.mfscFeatSz();
  } else if (FLAGS_mfcc) {
    numFeatures = featparams.mfccFeatSz();
  }
  return numFeatures;
}

} // namespace w2l

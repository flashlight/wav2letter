/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <arrayfire.h>

#include "recipes/models/local_prior_match/src/runtime/Utils.h"

#include "common/Defines.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "recipes/models/local_prior_match/src/runtime/Defines.h"

namespace w2l {

std::vector<int> genTokenDictIndexMap(
    const Dictionary& dict1,
    const Dictionary& dict2) {
  int size1 = dict1.indexSize();

  std::vector<int> mapping(size1);
  for (int idx1 = 0; idx1 < size1; ++idx1) {
    auto token = dict1.getEntry(idx1);
    // assume we already ran `dict2.setDefaultIndex(...);`
    auto idx2 = dict2.getIndex(token);
    mapping[idx1] = idx2;
  }

  return mapping;
}

af::array getTargetLength(af::array& target, int eosIdx) {
  return af::sum(target != eosIdx, 0).as(af::dtype::s32) + 1;
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> batchBeamSearch(
    const fl::Variable& output,
    const std::shared_ptr<Seq2SeqCriterion>& criterion,
    int eos) {
  std::vector<std::vector<int>> paths;
  std::vector<int> hypoNums;
  for (int b = 0; b < output.dims(2); b++) {
    std::vector<Seq2SeqCriterion::CandidateHypo> initBeam;
    initBeam.emplace_back(Seq2SeqCriterion::CandidateHypo{});
    auto hypos = criterion->beamSearch(
        output.array()(af::span, af::span, b),
        initBeam,
        FLAGS_lpmBeamsz,
        FLAGS_maxdecoderoutputlen);

    for (auto& hypo : hypos) {
      hypo.path.push_back(eos);
      paths.push_back(hypo.path);
    }
    hypoNums.push_back(hypos.size());
  }

  return std::make_pair(paths, hypoNums);
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> filterBeamByLength(
    const std::vector<std::vector<int>>& paths,
    const std::vector<int>& hypoNums,
    const std::vector<int>& refLengths) {
  if (hypoNums.size() != refLengths.size()) {
    throw std::runtime_error(
        "size of hypoNums (" + std::to_string(hypoNums.size()) +
        ") and refLengths (" + std::to_string(refLengths.size()) +
        ") do not match");
  }

  if (FLAGS_hyplenratiolb < 0 && FLAGS_hyplenratioub < 0) {
    return std::make_pair(paths, hypoNums);
  }

  int offset = 0;
  std::vector<std::vector<int>> newPaths;
  std::vector<int> newHypoNums;
  for (int b = 0; b < hypoNums.size(); b++) {
    int newHypoNum = 0;
    int lb = std::floor(FLAGS_hyplenratiolb * refLengths[b]);
    int ub = std::ceil(FLAGS_hyplenratioub * refLengths[b]);

    for (int i = 0; i < hypoNums[b]; i++) {
      int curIdx = offset + i;
      auto curLen = paths[curIdx].size();
      // also remove length=1 (empty hypotheses)
      if (curLen <= 1 || (ub > 0 && curLen > ub) || curLen < lb) {
        continue;
      }
      newPaths.push_back(paths[curIdx]);
      newHypoNum += 1;
    }
    offset += hypoNums[b];
    newHypoNums.push_back(newHypoNum);
  }

  return std::make_pair(newPaths, newHypoNums);
}

af::array batchTarget(
    const std::vector<std::vector<int>>& tgt,
    const int& padVal) {
  std::vector<int> vecTgt;
  af::dim4 vecTgtDims;
  int batchSz = tgt.size();
  size_t maxTgtSize = 0;

  if (batchSz == 0) {
    return af::array();
  }

  for (const auto& t : tgt) {
    if (t.size() == 0) {
      throw std::runtime_error("Target has zero length.");
    }
    maxTgtSize = std::max(maxTgtSize, t.size());
  }
  // L X BATCHSZ (Col Major)
  vecTgt.resize(maxTgtSize * batchSz, padVal);
  vecTgtDims = af::dim4(maxTgtSize, batchSz);

  for (size_t i = 0; i < batchSz; ++i) {
    std::copy(tgt[i].begin(), tgt[i].end(), vecTgt.begin() + maxTgtSize * i);
  }
  return af::array(vecTgtDims, vecTgt.data());
}

fl::Variable batchEncoderOutput(
    const std::vector<int>& hypoNums,
    const fl::Variable& encoderOutput) {
  // tile and batch encoder outputs
  std::vector<fl::Variable> tiledEncoderOutputVec;
  for (int i = 0; i < hypoNums.size(); i++) {
    if (hypoNums[i] > 0) {
      auto curOutpt = encoderOutput(af::span, af::span, i);
      auto curTiledOutput = fl::tile(curOutpt, af::dim4(1, 1, hypoNums[i]));
      tiledEncoderOutputVec.emplace_back(curTiledOutput);
    }
  }
  return concatenate(tiledEncoderOutputVec, 2);
}

// maybe renormalize and/or change to linear scale
fl::Variable adjustProb(
    const fl::Variable& logprob,
    const std::vector<int>& hypoNums,
    bool renormalize,
    bool linear) {
  if (!renormalize && !linear) {
    return logprob;
  }

  std::vector<fl::Variable> outputVec;
  int offset = 0;
  for (auto& hypoNum : hypoNums) {
    if (hypoNum > 0) {
      auto logprobSlice = logprob(af::seq(offset, offset + hypoNum - 1));
      if (renormalize && linear) {
        outputVec.emplace_back(fl::softmax(logprobSlice, 0));
      } else if (renormalize && !linear) {
        outputVec.emplace_back(fl::logSoftmax(logprobSlice, 0));
      } else if (!renormalize && linear) {
        outputVec.emplace_back(fl::exp(logprobSlice));
      } else {
        throw std::runtime_error(
            "Something is really wrong. Should never arrive here.");
      }
      offset += hypoNum;
    }
  }
  if (offset != logprob.dims()[0]) {
    throw std::runtime_error(
        "Total number of hypos inconsistent : " + std::to_string(offset) +
        " vs " + std::to_string(logprob.dims()[0]));
  }
  return concatenate(outputVec, 0);
}

fl::Variable entropy(const fl::Variable& p) {
  fl::Variable logp = fl::log(p + 1e-6);
  fl::Variable ent = fl::sum(fl::negate(p * logp), {0});
  return ent;
}

} // namespace w2l

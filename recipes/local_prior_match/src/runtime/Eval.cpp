/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "recipes/models/local_prior_match/src/runtime/Eval.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "recipes/models/local_prior_match/src/runtime/Logging.h"

namespace w2l {

void evalOutput(
    const af::array& op,
    const af::array& target,
    std::map<std::string, fl::EditDistanceMeter>& mtr,
    const Dictionary& tgtDict,
    std::shared_ptr<SequenceCriterion> criterion) {
  auto batchsz = op.dims(2);
  for (int b = 0; b < batchsz; ++b) {
    auto tgt = target(af::span, b);
    auto viterbipath =
        afToVector<int>(criterion->viterbiPath(op(af::span, af::span, b)));
    auto tgtraw = afToVector<int>(tgt);

    remapLabels(viterbipath, tgtDict);
    remapLabels(tgtraw, tgtDict);

    auto ltrPred = tknPrediction2Ltr(viterbipath, tgtDict);
    auto ltrTgt = tknTarget2Ltr(tgtraw, tgtDict);
    auto wrdPred = tkn2Wrd(ltrPred);
    auto wrdTgt = tkn2Wrd(ltrTgt);

    mtr[kTarget].add(ltrPred, ltrTgt);
    mtr[kWord].add(wrdPred, wrdTgt);
  }
}

void evalDataset(
    std::shared_ptr<fl::Module> ntwrk,
    std::shared_ptr<SequenceCriterion> crit,
    std::shared_ptr<W2lDataset> testds,
    SSLDatasetMeters& mtrs,
    const Dictionary& dict) {
  resetDatasetMeters(mtrs);

  for (auto& sample : *testds) {
    auto output = ntwrk->forward({fl::input(sample[kInputIdx])}).front();
    auto critOut =
        crit->forward({output, fl::Variable(sample[kTargetIdx], false)});
    mtrs.values[kASRLoss].add(critOut[0].array());

    evalOutput(output.array(), sample[kTargetIdx], mtrs.edits, dict, crit);
  }
}

void runEval(
    std::shared_ptr<fl::Module> network,
    std::shared_ptr<SequenceCriterion> criterion,
    const std::unordered_map<std::string, std::shared_ptr<W2lDataset>>& ds,
    SSLTrainMeters& meters,
    const Dictionary& dict) {
  network->eval();
  criterion->eval();

  for (auto& d : ds) {
    evalDataset(network, criterion, d.second, meters.valid[d.first], dict);
  }
}

} // namespace w2l

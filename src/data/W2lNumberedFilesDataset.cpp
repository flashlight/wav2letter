/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "W2lNumberedFilesDataset.h"

#include <glog/logging.h>
#include <functional>
#include <numeric>

#include "common/Defines.h"
#include "common/Utils.h"

namespace w2l {

W2lNumberedFilesDataset::W2lNumberedFilesDataset(
    const std::string& paths,
    const DictionaryMap& dicts,
    int64_t batchsize /* = 1 */,
    int worldrank /* = 0 */,
    int worldsize /* = 1 */,
    const std::string& rootdir /* = "" */)
    : W2lDataset(dicts, batchsize, worldrank, worldsize),
      cumulativeSizes_({0}) {
  auto pathsVec = split(',', paths);
  TargetExtMap targetExts;
  targetExts.insert({kTargetIdx, FLAGS_target});
  if (!FLAGS_lexicon.empty()) {
    targetExts.insert({kWordIdx, "wrd"});
  }

  int64_t curSize = 0;
  for (const auto& p : pathsVec) {
    loaders_.emplace_back(
        pathsConcat(rootdir, trim(p)), FLAGS_input, targetExts);
    curSize += loaders_.back().size();
    cumulativeSizes_.emplace_back(curSize);
  }
  auto speechSamplesMetaInfo = loadSampleSizes();
  filterSamples(
      speechSamplesMetaInfo,
      FLAGS_minisz,
      FLAGS_maxisz,
      FLAGS_mintsz,
      FLAGS_maxtsz);
  sampleCount_ = speechSamplesMetaInfo.size();
  sampleSizeOrder_ = sortSamples(
      speechSamplesMetaInfo,
      FLAGS_dataorder,
      FLAGS_inputbinsize,
      FLAGS_outputbinsize);
  shuffle(-1);
  LOG(INFO) << "Total batches (i.e. iters): " << sampleBatches_.size();
}

W2lNumberedFilesDataset::~W2lNumberedFilesDataset() {
  threadpool_ = nullptr; // join all threads
}

std::vector<W2lLoaderData> W2lNumberedFilesDataset::getLoaderData(
    const int64_t idx) const {
  std::vector<W2lLoaderData> data;
  for (auto i : sampleBatches_[idx]) {
    int64_t permidx = sampleSizeOrder_[i];
    int64_t loaderidx =
        std::upper_bound(
            cumulativeSizes_.begin(), cumulativeSizes_.end(), permidx) -
        cumulativeSizes_.begin() - 1;
    int64_t fileidx = permidx - cumulativeSizes_[loaderidx];
    data.emplace_back(loaders_[loaderidx].get(fileidx));
  }
  return data;
}

std::vector<SpeechSampleMetaInfo> W2lNumberedFilesDataset::loadSampleSizes() {
  std::vector<SpeechSampleMetaInfo> speechSamplesMetaInfo(
      cumulativeSizes_.back());
  for (int64_t j = 0; j < loaders_.size(); ++j) {
    const auto& l = loaders_[j];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < l.size(); ++i) {
      auto audiofile = l.filename(i, FLAGS_input);
      auto targetfile = l.filename(i, FLAGS_target);
      auto idx = cumulativeSizes_[j] + i;
      auto info = w2l::loadSoundInfo(audiofile.c_str());
      auto durationMs =
          (static_cast<double>(info.frames) / info.samplerate) * 1e3;
      auto ref = loadTarget(targetfile);
      speechSamplesMetaInfo[idx] =
          SpeechSampleMetaInfo(durationMs, ref.size(), idx);
    }
  }
  return speechSamplesMetaInfo;
}
} // namespace w2l

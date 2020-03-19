/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "W2lDataset.h"

#include <functional>
#include <numeric>
#include <utility>

#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"

namespace w2l {

W2lDataset::W2lDataset(
    const DictionaryMap& dicts,
    int64_t batchsize,
    int worldrank /* = 0 */,
    int worldsize /* = 1 */)
    : dicts_(dicts),
      batchSize_(batchsize),
      worldRank_(worldrank),
      worldSize_(worldsize),
      threadpool_(fl::cpp::make_unique<fl::ThreadPool>(FLAGS_nthread)) {
  if (batchSize_ < 1 || worldRank_ < 0 || worldSize_ < 1 ||
      worldRank_ >= worldSize_) {
    LOG(FATAL) << "Invalid arguments!";
  }
}

int64_t W2lDataset::size() const {
  return sampleBatches_.size();
}

std::vector<af::array> W2lDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  W2lFeatureData feat;
  if (FLAGS_nthread > 0) {
    feat = getFeatureDataAndPrefetch(idx);
  } else {
    feat = getFeatureData(idx);
  }
  std::vector<af::array> result(kNumDataIdx);
  result[kInputIdx] = feat.input.empty()
      ? af::array(feat.inputDims)
      : af::array(feat.inputDims, feat.input.data());
  for (const auto& target : feat.targets) {
    auto targetType = target.first;
    auto targetData = target.second;
    auto targetDims = feat.targetDims[targetType];
    result[targetType] = targetData.empty()
        ? af::array(targetDims)
        : af::array(targetDims, targetData.data());
  }
  result[kSampleIdx] = feat.sampleIds.empty()
      ? af::array(feat.sampleIdsDims)
      : af::array(feat.sampleIdsDims, feat.sampleIds.data());
  return result;
}

int64_t W2lDataset::getGlobalBatchIdx(const int64_t idx) {
  return sampleBatches_[idx][0] / (worldSize_ * batchSize_);
}

W2lFeatureData W2lDataset::getFeatureData(const int64_t idx) const {
  auto ldData = getLoaderData(idx);
  return featurize(ldData, dicts_);
}

W2lFeatureData W2lDataset::getFeatureDataAndPrefetch(const int64_t idx) const {
  W2lFeatureData feat;
  // check cache
  auto cachedata = prefetchCache_.find(idx);
  if (cachedata != prefetchCache_.end()) {
    feat = cachedata->second.get();
    prefetchCache_.erase(idx);
  } else {
    feat = getFeatureData(idx);
  }

  int64_t prefetchSize = FLAGS_nthread;
  // remove from cache (if necessary)
  for (auto it = prefetchCache_.begin(); it != prefetchCache_.end();) {
    if (it->first < idx || it->first > idx + prefetchSize) {
      it = prefetchCache_.erase(it);
      continue;
    } else {
      ++it;
    }
  }

  // add to cache
  for (int64_t i = idx + 1; i < std::min(idx + 1 + prefetchSize, size()); ++i) {
    if (prefetchCache_.find(i) == prefetchCache_.end()) {
      prefetchCache_.emplace(
          i,
          threadpool_->enqueue(
              [this](int64_t j) { return this->getFeatureData(j); }, i));
    }
  }
  return feat;
}

void W2lDataset::shuffle(int seed) {
  prefetchCache_.clear();
  RoundRobinBatchPacker shuffler(batchSize_, worldSize_, worldRank_);
  // We shuffle such that calling `get(idx)` from different mpi jobs with same
  // `idx` would return similar length samples
  sampleBatches_ = shuffler.getBatches(sampleCount_, seed);
}

std::vector<std::vector<int64_t>> RoundRobinBatchPacker::getBatches(
    int64_t nSamples,
    int64_t seed) const {
  // Randomly shuffle the global batch ids
  // global batch is the batch containing all utterances which
  // are processed in 1 iteration

  int64_t nSamplesPerGlobalBatch = worldSize_ * batchSize_;

  int64_t nGlobalBatches = nSamples / nSamplesPerGlobalBatch;

  // Include-last if we can fit atleast one sample for last batch for all ranks
  bool includeLast = (nSamples % nSamplesPerGlobalBatch) >= worldSize_;

  if (includeLast) {
    ++nGlobalBatches;
  }

  std::vector<int64_t> globalBatchIdx(nGlobalBatches);
  std::iota(globalBatchIdx.begin(), globalBatchIdx.end(), 0);

  if (seed >= 0) {
    auto rng = std::mt19937(seed);
    auto n = globalBatchIdx.size();
    // custom implementation of shuffle - https://stackoverflow.com/a/51931164
    for (auto i = n; i >= 1; --i) {
      std::swap(globalBatchIdx[i - 1], globalBatchIdx[rng() % n]);
    }
  }

  std::vector<std::vector<int64_t>> batches(nGlobalBatches);
  for (size_t i = 0; i < nGlobalBatches; i++) {
    auto offset = globalBatchIdx[i] * nSamplesPerGlobalBatch;
    int64_t nCurSamples; // num samples in current batch
    if (includeLast && (globalBatchIdx[i] == nGlobalBatches - 1)) {
      nCurSamples = (nSamples - offset) / worldSize_; // min samples per proc
      int64_t remaining = (nSamples - offset) % worldSize_;
      offset += nCurSamples * worldRank_;
      if (worldRank_ < remaining) {
        nCurSamples += 1;
      }
      offset += std::min(worldRank_, remaining);
    } else {
      offset += batchSize_ * worldRank_;
      nCurSamples = batchSize_;
    }
    std::vector<std::int64_t> curBatch(nCurSamples);
    std::iota(curBatch.begin(), curBatch.end(), offset);
    batches[i] = std::move(curBatch);
  }
  return batches;
}

} // namespace w2l

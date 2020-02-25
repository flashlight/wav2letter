/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "recipes/models/local_prior_match/src/runtime/DataScheduler.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "recipes/models/local_prior_match/src/runtime/Defines.h"
#include "runtime/Logger.h"

namespace w2l {

DataScheduler::DataScheduler(
    const std::vector<std::shared_ptr<W2lDataset>>& datasets,
    const std::vector<int64_t>& dataTypes,
    const std::vector<int64_t>& numIters,
    int64_t curEpoch /* = 1 */)
    : ds_(datasets.begin(), datasets.end()),
      dataTypes_(dataTypes.begin(), dataTypes.end()),
      dsNumIters_(numIters.begin(), numIters.end()),
      dsCurIter_(ds_.size(), 0),
      dsIterOffset_(ds_.size(), 0),
      dsCurEpochs_(ds_.size(), curEpoch),
      gen_(FLAGS_seed) {
  LOG_IF(FATAL, datasets.size() == 0) << "No datasets to be added";
  LOG_IF(FATAL, ds_.size() != dataTypes_.size())
      << "mismatch between the number of datasets "
      << "and the number data types specified";

  if (!FLAGS_noresample) {
    for (auto& d : ds_) {
      LOG_MASTER(INFO) << "Shuffling trainset";
      d->shuffle(curEpoch);
    }
  }
  initialize();
}

void DataScheduler::initialize() {
  LOG_IF(FATAL, ds_.size() != dsNumIters_.size())
      << "mismatch between the number of datasets "
      << "and the number of schedules specified";

  dsCumNumIters_.resize(dsNumIters_.size());
  for (int i = 0; i < dsNumIters_.size(); ++i) {
    LOG_IF(FATAL, dsNumIters_[i] < 0)
        << "Invalid training schedule (number of iterations < 0)";
    if (i == 0) {
      dsCumNumIters_[i] = dsNumIters_[i];
    } else {
      dsCumNumIters_[i] = dsNumIters_[i] + dsCumNumIters_[i - 1];
    }
  }
  LOG_IF(FATAL, dsCumNumIters_.back() == 0)
      << "Invalid training schedule (zero iterations on all datasets)";

  if (FLAGS_schedulerorder == kInOrder) {
    curDs_ = 0;
    while (curDs_ < dsNumIters_.size() && dsNumIters_[curDs_] == 0) {
      ++curDs_;
    }
  } else if (FLAGS_schedulerorder == kUniformOrder) {
    curDs_ = std::max_element(dsNumIters_.begin(), dsNumIters_.end()) -
        dsNumIters_.begin();
  } else if (FLAGS_schedulerorder == kRandomOrder) {
    std::uniform_int_distribution<int> distribution(1, dsCumNumIters_.back());
    auto d = distribution(gen_);
    auto lit =
        std::lower_bound(dsCumNumIters_.begin(), dsCumNumIters_.end(), d);
    curDs_ = lit - dsCumNumIters_.begin();
  } else {
    LOG(FATAL) << "unimplemented order: " << FLAGS_schedulerorder;
  }
}

std::vector<af::array> DataScheduler::get() {
  auto idx = (dsIterOffset_[curDs_] + dsCurIter_[curDs_]) % ds_[curDs_]->size();
  auto sample = ds_[curDs_]->get(idx);
  auto globalBatchIdx = ds_[curDs_]->getGlobalBatchIdx(idx);
  sample.emplace_back(af::constant(dataTypes_[curDs_], 1, s64));
  sample.emplace_back(af::constant(globalBatchIdx, 1, s64));

  update();
  return sample;
}

void DataScheduler::update() {
  ++dsCurIter_[curDs_];

  if (!FLAGS_noresample &&
      (dsIterOffset_[curDs_] + dsCurIter_[curDs_]) % ds_[curDs_]->size() == 0) {
    LOG_MASTER(INFO) << "Shuffling trainset";
    ds_[curDs_]->shuffle(++dsCurEpochs_[curDs_] /* seed */);
  }

  if (FLAGS_schedulerorder == kInOrder) {
    if (dsCurIter_[curDs_] % dsNumIters_[curDs_] == 0) {
      curDs_ = (curDs_ + 1) % ds_.size();
      while (dsNumIters_[curDs_] == 0) {
        curDs_ = (curDs_ + 1) % ds_.size();
      }
    }
  } else if (FLAGS_schedulerorder == kUniformOrder) {
    double minVal = std::numeric_limits<double>::max();
    for (int i = 0; i < ds_.size(); ++i) {
      if (dsNumIters_[i] > 0) {
        int offset = dsCurIter_[i] / dsNumIters_[i];
        double ratio =
            1.0 / (dsNumIters_[i] + 1) * (dsCurIter_[i] % dsNumIters_[i] + 1);
        if (offset + ratio < minVal) {
          minVal = offset + ratio;
          curDs_ = i;
        }
      }
    }
  } else if (FLAGS_schedulerorder == kRandomOrder) {
    for (int c = curDs_; c < dsCumNumIters_.size(); ++c) {
      --dsCumNumIters_[c];
    }
    if (dsCumNumIters_.back() == 0) {
      std::partial_sum(
          dsNumIters_.begin(), dsNumIters_.end(), dsCumNumIters_.begin());
    }
    std::uniform_int_distribution<int> distribution(1, dsCumNumIters_.back());
    auto d = distribution(gen_);
    auto lit =
        std::lower_bound(dsCumNumIters_.begin(), dsCumNumIters_.end(), d);
    curDs_ = lit - dsCumNumIters_.begin();
  }
}

std::vector<int64_t> DataScheduler::getSchedule() {
  return dsNumIters_;
}

void DataScheduler::setSchedule(std::vector<int64_t> newIters) {
  dsNumIters_ = std::move(newIters);
  initialize();
  for (int i = 0; i < dsCurIter_.size(); ++i) {
    dsIterOffset_[i] = (dsIterOffset_[i] + dsCurIter_[i]) % ds_[i]->size();
    dsCurIter_[i] = 0;
  }
}
} // namespace w2l

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <flashlight/flashlight.h>

#include "data/W2lDataset.h"

namespace w2l {

class DataScheduler {
 public:
  /** The `DataScheduler` class constructor.
   * @param datasets Pointers to the datasets.
   * @param dataTypes Each dataset's type (kParallelData or kUnpairedAudio).
   * @param numIters Number of iterations on each dataset before switching to
   * the next dataset.
   * @param curEpoch Number of epochs that the datasets have been iterated
   * through for dataset shuffling use.
   */
  DataScheduler(
      const std::vector<std::shared_ptr<W2lDataset>>& datasets,
      const std::vector<int64_t>& dataTypes,
      const std::vector<int64_t>& numIters,
      int64_t curEpoch = 1);

  // sequentially access the data according to the schedule
  std::vector<af::array> get();

  std::vector<int64_t> getSchedule();

  void setSchedule(std::vector<int64_t> newIters);

 private:
  std::vector<std::shared_ptr<W2lDataset>> ds_;
  std::vector<int64_t> dataTypes_;
  std::vector<int64_t> dsNumIters_;
  std::vector<int64_t> dsCumNumIters_;
  std::vector<int64_t> dsCurIter_;
  std::vector<int64_t> dsIterOffset_;
  std::vector<int64_t> dsCurEpochs_;
  size_t curDs_;

  std::mt19937 gen_;

  DataScheduler() {}

  void initialize();

  void update();
};

} // namespace w2l

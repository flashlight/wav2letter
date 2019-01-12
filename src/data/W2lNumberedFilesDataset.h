/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "data/NumberedFilesLoader.h"
#include "data/Utils.h"
#include "data/W2lDataset.h"

namespace w2l {

class W2lNumberedFilesDataset : public W2lDataset {
 public:
  W2lNumberedFilesDataset(
      const std::string& paths,
      const DictionaryMap& dicts,
      int64_t batchsize,
      int worldrank = 0,
      int worldsize = 1,
      const std::string& rootdir = "");

  ~W2lNumberedFilesDataset() override;

  virtual std::vector<W2lLoaderData> getLoaderData(
      const int64_t idx) const override;

 private:
  std::vector<NumberedFilesLoader> loaders_;
  std::vector<int64_t> cumulativeSizes_;
  std::vector<int64_t> sampleSizeOrder_;

  std::vector<SpeechSampleMetaInfo> loadSampleSizes();
};
} // namespace w2l

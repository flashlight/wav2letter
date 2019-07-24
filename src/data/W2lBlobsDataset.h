/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "common/Utils.h"
#include "data/Utils.h"
#include "data/W2lDataset.h"

namespace w2l {

class W2lBlobsDataset : public W2lDataset {
 public:
  W2lBlobsDataset(
      const std::string& filenames,
      const DictionaryMap& dicts,
      const LexiconMap& lexicon,
      int64_t batchSize,
      int worldRank = 0,
      int worldSize = 1,
      bool fallback2Ltr = false,
      bool skipUnk = false,
      const std::string& rootdir = "");

  ~W2lBlobsDataset() override;

  virtual std::vector<W2lLoaderData> getLoaderData(
      const int64_t idx) const override;

 private:
  std::vector<std::shared_ptr<fl::BlobDataset>> blobs_;
  std::vector<int64_t> sampleSizeOrder_;
  std::vector<int64_t> blobIndex_;
  std::vector<int64_t> sampleIndex_;
  LexiconMap lexicon_;
  bool includeWrd_;
  bool fallback2Ltr_;
  bool skipUnk_;

  std::vector<SpeechSampleMetaInfo> loadBlob(const std::string& filename);
};
} // namespace w2l

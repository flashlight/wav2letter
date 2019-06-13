/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <string>

#include "common/Dictionary.h"
#include "data/Sound.h"

namespace w2l {

typedef std::unordered_map<int, std::string> TargetExtMap;
typedef std::unordered_map<int, std::vector<std::string>> TargetMap;

struct W2lLoaderData {
  std::vector<float> input;
  TargetMap targets;
  std::string sampleId;
};

// class NumberedFilesLoader
//
// The files are named in the following way
//    00000/000000000.abc, 00000/000000001.abc if hasSubdir_ = true
//    000000000.abc, 000000001.abc if hasSubdir_ = false
class NumberedFilesLoader {
 public:
  NumberedFilesLoader(
      const std::string& path,
      const std::string& inputext,
      const TargetExtMap& targetexts);

  int64_t size() const;

  W2lLoaderData get(const int64_t idx) const;

  std::string filename(int64_t idx, const std::string& extension) const;

 private:
  const std::string path_;
  bool hasSubdir_;
  int64_t startIdx_;
  int64_t size_;
  std::string inputExtension_;
  TargetExtMap targetExtensions_;
};
} // namespace w2l

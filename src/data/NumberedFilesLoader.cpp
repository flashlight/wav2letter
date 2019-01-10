/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "NumberedFilesLoader.h"

#include <algorithm>
#include <array>
#include <string>

#include <glog/logging.h>

#include "common/Utils.h"

namespace w2l {

using namespace fl;

NumberedFilesLoader::NumberedFilesLoader(
    const std::string& path,
    const std::string& inputext,
    const TargetExtMap& targetexts)
    : path_(path), inputExtension_(inputext), targetExtensions_(targetexts) {
  auto testkey = inputExtension_;
  LOG(INFO) << "Adding dataset " << path << " ...";

  if (!dirExists(path)) {
    LOG(FATAL) << "Directory '" << path << "' doesn't exist";
  }

  // Check if dataset is in one of the following allowed formats
  // 1. 000000000.abc ... (hasSubdir_ = false, startIdx_ can be 0 or 1)
  // 2. 00000/000000000.abc ... (hasSubdir_= true, startIdx_ = 0)
  std::string file0, file1, file2;
  startIdx_ = 0;
  hasSubdir_ = false;
  file0 = filename(0, testkey);
  if (!fileExists(file0)) {
    startIdx_ = 1;
    file1 = filename(0, testkey);
    if (!fileExists(file1)) {
      startIdx_ = 0;
      hasSubdir_ = true;
      file2 = filename(0, testkey);
      if (!fileExists(file2)) {
        LOG(FATAL) << "Invalid dataset path. No file found - <" << file0
                   << "> nor in <" << file1 << "> nor in <" << file2 << ">";
      }
    }
  }

  // Use binary search to find the size of the dataset.
  // File numbers are assumed to be contiguous.
  int64_t lo = 0, hi = 999999999;
  while (hi != lo) {
    auto mid = lo + ((hi - lo) >> 1);
    if (fileExists(filename(mid, testkey))) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  size_ = lo;
  LOG(INFO) << size_ << " files found. ";
}

W2lLoaderData NumberedFilesLoader::get(const int64_t idx) const {
  if (!(idx >= 0 && idx < size())) {
    throw std::out_of_range("NumberedFilesLoader idx out of range");
  }
  W2lLoaderData data;
  auto inputpath = filename(idx, inputExtension_);
  data.sampleId = std::to_string(idx);
  data.input = speech::loadSound<float>(inputpath.c_str());

  for (auto& targetExtension : targetExtensions_) {
    auto targetpath = filename(idx, targetExtension.second);
    data.targets[targetExtension.first] = loadTarget(targetpath);
  }
  return data;
}

std::string NumberedFilesLoader::filename(
    int64_t idx,
    const std::string& extension) const {
  std::string name = path_;
  std::array<char, 20> fchar;
  if (hasSubdir_) {
    snprintf(fchar.data(), fchar.size(), "%05ld", idx / 10000);
    name = pathsConcat(name, std::string(fchar.data()));
  }
  snprintf(fchar.data(), fchar.size(), "%09ld.", idx + startIdx_);
  return pathsConcat(name, std::string(fchar.data()) + extension);
}

int64_t NumberedFilesLoader::size() const {
  return size_;
}
} // namespace w2l

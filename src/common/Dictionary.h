/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace w2l {

// A simple dictionary class which holds a bidirectional map
// tokens (strings) <--> integer indices. Not thread-safe !
class Dictionary {
 public:
  Dictionary() {}

  size_t tokenSize() const;

  size_t indexSize() const;

  void addToken(const std::string& token, int idx);

  void addToken(const std::string& token);

  std::string getToken(int idx) const;

  void setDefaultIndex(int idx);

  int getIndex(const std::string& token) const;

  bool contains(const std::string& token) const;

  // checks if all the indices are contiguous
  bool isContiguous() const;

  std::vector<int> mapTokensToIndices(
      const std::vector<std::string>& tokens) const;

  std::vector<std::string> mapIndicesToTokens(
      const std::vector<int>& indices) const;

 private:
  std::unordered_map<std::string, int> token2idx_;
  std::unordered_map<int, std::string> idx2token_;
  int defaultIndex_ = -1;
};

typedef std::unordered_map<int, Dictionary> DictionaryMap;

} // namespace w2l

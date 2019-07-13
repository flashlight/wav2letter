/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

namespace w2l {

// A simple dictionary class which holds a bidirectional map
// entry (strings) <--> integer indices. Not thread-safe !
class Dictionary {
 public:
  // Creates an empty dictionary
  Dictionary() {}

  explicit Dictionary(std::istream& stream);

  explicit Dictionary(const std::string& filename);

  size_t entrySize() const;

  size_t indexSize() const;

  void addEntry(const std::string& entry, int idx);

  void addEntry(const std::string& entry);

  std::string getEntry(int idx) const;

  void setDefaultIndex(int idx);

  int getIndex(const std::string& entry) const;

  bool contains(const std::string& entry) const;

  // checks if all the indices are contiguous
  bool isContiguous() const;

  std::vector<int> mapEntriesToIndices(
      const std::vector<std::string>& entries) const;

  std::vector<std::string> mapIndicesToEntries(
      const std::vector<int>& indices) const;

 private:
  // Creates a dictionary from an input stream
  void createFromStream(std::istream& stream);

  std::unordered_map<std::string, int> entry2idx_;
  std::unordered_map<int, std::string> idx2entry_;
  int defaultIndex_ = -1;
};

typedef std::unordered_map<int, Dictionary> DictionaryMap;

} // namespace w2l

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Dictionary.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace w2l {

void Dictionary::addToken(const std::string& token, int idx) {
  if (token2idx_.find(token) != token2idx_.end()) {
    throw std::invalid_argument("Duplicate entry name in dictionary: " + token);
  }
  token2idx_[token] = idx;
  if (idx2token_.find(idx) == idx2token_.end()) {
    idx2token_[idx] = token;
  }
}

void Dictionary::addToken(const std::string& token) {
  int idx = idx2token_.size();
  // Find first available index.
  while (idx2token_.find(idx) != idx2token_.end()) {
    ++idx;
  }
  addToken(token, idx);
}

std::string Dictionary::getToken(int idx) const {
  auto iter = idx2token_.find(idx);
  if (iter == idx2token_.end()) {
    throw std::invalid_argument(
        "Unknown index in dictionary: " + std::to_string(idx));
  }
  return iter->second;
}

void Dictionary::setDefaultIndex(int idx) {
  defaultIndex_ = idx;
}

int Dictionary::getIndex(const std::string& token) const {
  auto iter = token2idx_.find(token);
  if (iter == token2idx_.end()) {
    if (defaultIndex_ < 0) {
      throw std::invalid_argument("Unknown token in dictionary: " + token);
    } else {
      std::cerr << "Skipping unknown token: " << token << "\n";
      return defaultIndex_;
    }
  }
  return iter->second;
}

bool Dictionary::contains(const std::string& token) const {
  auto iter = token2idx_.find(token);
  if (iter == token2idx_.end()) {
    return false;
  }
  return true;
}

size_t Dictionary::tokenSize() const {
  return token2idx_.size();
}

bool Dictionary::isContiguous() const {
  for (size_t i = 0; i < indexSize(); ++i) {
    if (idx2token_.find(i) == idx2token_.end()) {
      return false;
    }
  }
  for (const auto& tknidx : token2idx_) {
    if (idx2token_.find(tknidx.second) == idx2token_.end()) {
      return false;
    }
  }
  return true;
}

std::vector<int> Dictionary::mapTokensToIndices(
    const std::vector<std::string>& tokens) const {
  std::vector<int> indices;
  indices.reserve(tokens.size());
  for (const auto& tkn : tokens) {
    indices.emplace_back(getIndex(tkn));
  }
  return indices;
}

std::vector<std::string> Dictionary::mapIndicesToTokens(
    const std::vector<int>& indices) const {
  std::vector<std::string> tokens;
  tokens.reserve(indices.size());
  for (const auto& idx : indices) {
    tokens.emplace_back(getToken(idx));
  }
  return tokens;
}

size_t Dictionary::indexSize() const {
  return idx2token_.size();
}

} // namespace w2l

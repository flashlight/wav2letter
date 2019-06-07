/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <math.h>
#include <stdlib.h>
#include <limits>

#include "decoder/Trie.h"

namespace w2l {

const double kMinusLogThreshold = -39.14;

TrieNodePtr Trie::getRoot() {
  return root_;
}

int Trie::getNumChildren() {
  return maxChildren_;
}

TrieNodePtr
Trie::insert(const std::vector<int>& indices, int label, float score) {
  TrieNodePtr node = root_;
  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    if (idx < 0 || idx >= maxChildren_) {
      LOG(FATAL) << "[Trie] Invalid letter index: " << idx;
    }
    if (node->children_.find(idx) == node->children_.end()) {
      node->children_[idx] = std::make_shared<TrieNode>(idx);
    }
    node = node->children_[idx];
  }
  if (node->nLabel_ < kTrieMaxLabel) {
    node->label_[node->nLabel_] = label;
    node->score_[node->nLabel_] = score;
    node->nLabel_++;
  } else {
    LOG(INFO) << "[Trie] Trie label number reached limit: " << kTrieMaxLabel;
  }
  return node;
}

TrieNodePtr Trie::search(const std::vector<int>& indices) {
  TrieNodePtr node = root_;
  for (auto idx : indices) {
    if (idx < 0 || idx >= maxChildren_) {
      LOG(FATAL) << "[Trie] Invalid letter index: " << idx;
    }
    if (node->children_.find(idx) == node->children_.end()) {
      return nullptr;
    }
    node = node->children_[idx];
  }
  return node;
}

/* logadd */
double TrieLogAdd(double log_a, double log_b) {
  double minusdif;
  if (log_a < log_b) {
    std::swap(log_a, log_b);
  }
  minusdif = log_b - log_a;
  if (minusdif < kMinusLogThreshold) {
    return log_a;
  } else {
    return log_a + log1p(exp(minusdif));
  }
}

void smearNode(TrieNodePtr node, SmearingMode smearMode) {
  node->maxScore_ = -std::numeric_limits<float>::infinity();
  for (int idx = 0; idx < node->nLabel_; idx++) {
    node->maxScore_ = TrieLogAdd(node->maxScore_, node->score_[idx]);
  }
  for (auto child : node->children_) {
    auto childNode = child.second;
    smearNode(childNode, smearMode);
    if (smearMode == SmearingMode::LOGADD) {
      node->maxScore_ = TrieLogAdd(node->maxScore_, childNode->maxScore_);
    } else if (
        smearMode == SmearingMode::MAX &&
        childNode->maxScore_ > node->maxScore_) {
      node->maxScore_ = childNode->maxScore_;
    }
  }
}

void Trie::smear(SmearingMode smearMode) {
  if (smearMode != SmearingMode::NONE) {
    smearNode(root_, smearMode);
  }
}

} // namespace w2l

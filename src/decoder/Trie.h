/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

namespace w2l {

const int kTrieMaxLabel = 6;

enum class SmearingMode {
  NONE = 0,
  MAX = 1,
  LOGADD = 2,
};

/**
 * TrieNode is the trie node structure in Trie.
 */
struct TrieNode {
  explicit TrieNode(int idx)
      : children_(std::unordered_map<int, std::shared_ptr<TrieNode>>()),
        idx_(idx),
        nLabel_(0),
        label_(kTrieMaxLabel),
        score_(kTrieMaxLabel),
        maxScore_(0) {}

  // Pointers to the childern of a node
  std::unordered_map<int, std::shared_ptr<TrieNode>> children_;

  // Node index
  int idx_;

  // Number of labels a node has. Note that nLabel_ is positive
  // only if the current code represent a completed token.
  int nLabel_;

  // Labels of words that are constructed from the given path
  std::vector<int> label_;

  // Scores (score_ should have the same size as label_)
  std::vector<float> score_;

  // Maximum score of all the labels if this node is a leaf,
  // otherwise it will be the value after trie smearing.
  float maxScore_;
};

typedef std::shared_ptr<TrieNode> TrieNodePtr;

/**
 * Trie is used to store the lexicon in langiage model. We use it to limit
 * the search space in deocder and quickly look up scores for a given token
 * (completed word) or make prediction for incompleted ones based on smearing.
 */
class Trie {
 public:
  Trie(int maxChildren, int rootIdx)
      : root_(std::make_shared<TrieNode>(rootIdx)), maxChildren_(maxChildren) {}

  /* Return the root node pointer */
  TrieNodePtr getRoot();

  /* Returns the number of childern for a given lexicon */
  int getNumChildren();

  /* Insert a token into trie with label */
  TrieNodePtr insert(const std::vector<int>& indices, int label, float score);

  /* Get the labels for a given token */
  TrieNodePtr search(const std::vector<int>& indices);

  /**
   * Smearing the trie using the valid labels inserted in the trie so as to get
   * score on each node (incompleted token).
   * For example, if smear_mode is MAX, then for node "a" in path "c"->"a", we
   * will select the maximum score from all its children like "c"->"a"->"t",
   * "c"->"a"->"n", "c"->"a"->"r"->"e" and so on.
   * This process will be carry out recusively on all the nodes.
   */
  void smear(const SmearingMode smear_mode);

 private:
  TrieNodePtr root_;
  int maxChildren_; // The maximum number of childern for each node. It is
                    // usually the size of letters or phonmes.
};

typedef std::shared_ptr<Trie> TriePtr;

} // namespace w2l

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

const int kTrieMaxLable = 6;

enum class SmearingMode {
  NONE = 0,
  MAX = 1,
  LOGADD = 2,
};

/**
 * TrieLabel is the label for trie nodes representing completed tokens. It
 * has two indices of each token, one from the dictionary in language model and
 * the other from the dictionary in decoder.
 */
struct TrieLabel {
  TrieLabel(int lm, int usr) : lm_(lm), usr_(usr) {}
  int lm_;
  int usr_;
};

typedef std::shared_ptr<TrieLabel> TrieLabelPtr;

/**
 * TrieNode is the trie node structure in Trie.
 */
struct TrieNode {
  TrieNode(int nchildren, int idx)
      : children_(std::unordered_map<int, std::shared_ptr<TrieNode>>()),
        idx_(idx),
        nLabel_(0),
        label_(std::vector<TrieLabelPtr>(kTrieMaxLable)),
        score_(std::vector<float>(kTrieMaxLable)),
        maxScore_(0) {}

  std::unordered_map<int, std::shared_ptr<TrieNode>>
      children_; // Pointers to the childern of a node
  int idx_; // Node index
  int nLabel_; // Number of labels a node has. Note that nLabel_ is positive
               // only if the current code represent a completed token.
  std::vector<TrieLabelPtr> label_; // Labels
  std::vector<float>
      score_; // Scores (score_ should have the same size as label_)
  float maxScore_; // Maximum score of all the labels if this node is a leaf,
                   // otherwise it will be the value after trie smearing.
};

typedef std::shared_ptr<TrieNode> TrieNodePtr;

/**
 * Trie is used to store the lexicon in langiage model. We use it to limit
 * the search space in deocder and quickly look up scores for a given token
 * (completed word) or make prediction for incompleted ones based on smearing.
 */
class Trie {
 public:
  Trie(int nChildren, int rootIdx)
      : root_(std::make_shared<TrieNode>(nChildren, rootIdx)),
        nChildren_(nChildren) {}

  /* Return the root node pointer */
  TrieNodePtr getRoot();

  /* Returns the number of childern for a given lexicon */
  int getNumChildren();

  /* Insert a token into trie with label */
  TrieNodePtr insert(
      const std::vector<int>& indices,
      const TrieLabelPtr label,
      float score);

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
  int nChildren_; // The maximum number of childern for each node. It is
                  // usually the size of letters or phonmes.
};

typedef std::shared_ptr<Trie> TriePtr;

} // namespace w2l

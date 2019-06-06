/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "decoder/Trie.h"
#include "decoder/WordLMDecoder.h"
#include "lm/KenLM.h"
#include "module/module.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"

using namespace w2l;

/**
 * In this test, we check the output from LM, trie and decoder.
 * T, N, emissions, transitions are randomly generated.
 * Letters and words are commonly used ones in our pipeline.
 * Language model is downloaded from Librispeech website:
 * http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz
 * We pruned it so as to have much smaller size.
 */

std::vector<int> tokens2Tensor(
    const std::string& spelling,
    const Dictionary& tokenDict) {
  std::vector<int> ret;
  ret.reserve(spelling.size());
  auto tokens = splitWrd(spelling);
  for (const auto& tkn : tokens) {
    ret.push_back(tokenDict.getIndex(tkn));
  }
  replaceReplabels(ret, FLAGS_replabel, tokenDict);
  return ret;
}

TEST(DecoderTest, run) {
  FLAGS_criterion = kAsgCriterion;
  FLAGS_replabel = 1;
  std::string dataDir = "";
#ifdef DECODER_TEST_DATADIR
  dataDir = DECODER_TEST_DATADIR;
#endif

  /* ===================== Create Dataset ===================== */
  EmissionSet emission_set;

  // T, N
  std::string tn_path = pathsConcat(dataDir, "TN.bin");
  std::ifstream tn_stream(tn_path, std::ios::binary | std::ios::in);
  std::vector<int> tn_array(2);
  int T, N;
  tn_stream.read((char*)tn_array.data(), 2 * sizeof(int));
  T = tn_array[0];
  N = tn_array[1];
  emission_set.emissionN = N;
  emission_set.emissionT.push_back(T);
  tn_stream.close();

  // Emission
  emission_set.emissions.resize(1);
  emission_set.emissions[0].resize(T * N);
  std::string emission_path = pathsConcat(dataDir, "emission.bin");
  std::ifstream em_stream(emission_path, std::ios::binary | std::ios::in);
  em_stream.read(
      (char*)emission_set.emissions[0].data(), T * N * sizeof(float));
  em_stream.close();

  // Transitions
  std::vector<float> transitions(N * N);
  std::string transitions_path = pathsConcat(dataDir, "transition.bin");
  std::ifstream tr_stream(transitions_path, std::ios::binary | std::ios::in);
  tr_stream.read((char*)transitions.data(), N * N * sizeof(float));
  tr_stream.close();

  LOG(INFO) << "[Serialization] Loaded emissions [" << T << " x " << N << ']';

  /* ===================== Create Dictionary ===================== */
  auto lexicon = loadWords(pathsConcat(dataDir, "words.lst"), -1);
  auto tokenDict = createTokenDict(pathsConcat(dataDir, "letters.lst"));
  auto wordDict = createWordDict(lexicon);

  LOG(INFO) << "[Dictionary] Number of words: " << wordDict.indexSize();

  /* ===================== Decode ===================== */
  /* -------- Build Language Model --------*/
  auto lm = std::make_shared<KenLM>(pathsConcat(dataDir, "lm.arpa"));
  LOG(INFO) << "[Decoder] LM constructed.\n";

  std::vector<std::string> sentence{"the", "cat", "sat", "on", "the", "mat"};
  auto inState = lm->start(0);
  float total_score = 0, lm_score = 0;
  std::vector<float> lmScoreTarget{
      -1.05971, -4.19448, -3.33383, -2.76726, -1.16237, -4.64589};
  for (int i = 0; i < sentence.size(); i++) {
    auto word = sentence[i];
    inState = lm->score(inState, lm->index(word), lm_score);
    ASSERT_NEAR(lm_score, lmScoreTarget[i], 1e-5);
    total_score += lm_score;
  }
  lm->finish(inState, lm_score);
  total_score += lm_score;
  ASSERT_NEAR(total_score, -19.5123, 1e-5);

  /* -------- Build Trie --------*/
  int sil_idx = tokenDict.getIndex(kSilToken);
  int blank_idx =
      FLAGS_criterion == kCtcCriterion ? tokenDict.getIndex(kBlankToken) : -1;
  int unk_idx = lm->index(kUnkToken);
  auto trie = std::make_shared<Trie>(tokenDict.indexSize(), sil_idx);
  auto start_state = lm->start(false);

  // Insert words
  for (auto& it : lexicon) {
    std::string word = it.first;
    int lm_idx = lm->index(word);
    if (lm_idx == unk_idx) {
      continue;
    }
    float score;
    auto dummy_state = lm->score(start_state, lm_idx, score);
    for (auto& spelling : it.second) {
      auto spelling_tensor = tkn2Idx(spelling, tokenDict);
      trie->insert(
          spelling_tensor,
          std::make_shared<TrieLabel>(lm_idx, wordDict.getIndex(word)),
          score);
    }
  }
  LOG(INFO) << "[Decoder] Trie planted.\n";

  // Smearing
  trie->smear(SmearingMode::MAX);
  LOG(INFO) << "[Decoder] Trie smeared.\n";

  std::vector<float> trieScoreTarget{
      -1.05971, -4.41062, -3.67099, -3.06203, -1.05971, -4.29683};
  for (int i = 0; i < sentence.size(); i++) {
    auto word = sentence[i];
    auto word_tensor = tokens2Tensor(word, tokenDict);
    auto node = trie->search(word_tensor);
    ASSERT_NEAR(node->maxScore_, trieScoreTarget[i], 1e-5);
  }

  /* -------- Build Decoder --------*/
  DecoderOptions decoder_opt(
      2500, // FLAGS_beamsize
      100.0, // FLAGS_beamthreshold
      2.0, // FLAGS_lmweight
      2.0, // FLAGS_lexiconcore
      -std::numeric_limits<float>::infinity(), // FLAGS_unkweight
      false, // FLAGS_logadd
      -1, // FLAGS_silweight
      CriterionType::ASG);

  std::shared_ptr<TrieLabel> unk =
      std::make_shared<TrieLabel>(unk_idx, wordDict.getIndex(kUnkToken));
  WordLMDecoder decoder(
      decoder_opt, trie, lm, sil_idx, blank_idx, unk, transitions);
  LOG(INFO) << "[Decoder] Decoder constructed.\n";

  /* -------- Run --------*/
  auto emission = emission_set.emissions[0];

  std::vector<float> score;
  std::vector<std::vector<int>> wordPredictions;
  std::vector<std::vector<int>> letterPredictions;

  auto timer = fl::TimeMeter();
  timer.resume();
  auto results = decoder.decode(emission.data(), T, N);
  timer.stop();

  int n_hyp = results.size();

  ASSERT_EQ(n_hyp, 877);

  std::vector<float> hypScoreTarget{
      -340.189, -340.415, -340.594, -340.653, -341.115};
  for (int i = 0; i < 5; i++) {
    ASSERT_NEAR(results[i].score_, hypScoreTarget[i], 1e-3);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

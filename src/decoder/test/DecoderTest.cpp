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
#include "runtime/runtime.h"

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
  std::string dataDir = "";
#ifdef DECODER_TEST_DATADIR
  dataDir = DECODER_TEST_DATADIR;
#endif

  /* ===================== Create Dataset ===================== */
  EmissionSet emissionSet;

  // T, N
  std::string tnPath = pathsConcat(dataDir, "TN.bin");
  std::ifstream tnStream(tnPath, std::ios::binary | std::ios::in);
  std::vector<int> tnArray(2);
  int T, N;
  tnStream.read((char*)tnArray.data(), 2 * sizeof(int));
  T = tnArray[0];
  N = tnArray[1];
  emissionSet.emissionN = N;
  emissionSet.emissionT.push_back(T);
  tnStream.close();

  // Emission
  emissionSet.emissions.resize(1);
  emissionSet.emissions[0].resize(T * N);
  std::string emissionPath = pathsConcat(dataDir, "emission.bin");
  std::ifstream em_stream(emissionPath, std::ios::binary | std::ios::in);
  em_stream.read((char*)emissionSet.emissions[0].data(), T * N * sizeof(float));
  em_stream.close();

  // Transitions
  std::vector<float> transitions(N * N);
  std::string transitionsPath = pathsConcat(dataDir, "transition.bin");
  std::ifstream tr_stream(transitionsPath, std::ios::binary | std::ios::in);
  tr_stream.read((char*)transitions.data(), N * N * sizeof(float));
  tr_stream.close();

  LOG(INFO) << "[Serialization] Loaded emissions [" << T << " x " << N << ']';

  /* ===================== Create Dictionary ===================== */
  auto lexicon = loadWords(pathsConcat(dataDir, "words.lst"), -1);
  Dictionary tokenDict(pathsConcat(dataDir, "letters.lst"));
  tokenDict.addEntry("1"); // replabel
  auto wordDict = createWordDict(lexicon);

  LOG(INFO) << "[Dictionary] Number of words: " << wordDict.indexSize();

  /* ===================== Decode ===================== */
  /* -------- Build Language Model --------*/
  auto lm = std::make_shared<KenLM>(pathsConcat(dataDir, "lm.arpa"), wordDict);
  LOG(INFO) << "[Decoder] LM constructed.\n";

  std::vector<std::string> sentence{"the", "cat", "sat", "on", "the", "mat"};
  auto inState = lm->start(0);
  float totalScore = 0, lmScore = 0;
  std::vector<float> lmScoreTarget{
      -1.05971, -4.19448, -3.33383, -2.76726, -1.16237, -4.64589};
  for (int i = 0; i < sentence.size(); i++) {
    const auto& word = sentence[i];
    std::tie(inState, lmScore) = lm->score(inState, wordDict.getIndex(word));
    ASSERT_NEAR(lmScore, lmScoreTarget[i], 1e-5);
    totalScore += lmScore;
  }
  std::tie(inState, lmScore) = lm->finish(inState);
  totalScore += lmScore;
  ASSERT_NEAR(totalScore, -19.5123, 1e-5);

  /* -------- Build Trie --------*/
  int silIdx = tokenDict.getIndex(kSilToken);
  int blankIdx =
      FLAGS_criterion == kCtcCriterion ? tokenDict.getIndex(kBlankToken) : -1;
  int unkIdx = wordDict.getIndex(kUnkToken);
  auto trie = std::make_shared<Trie>(tokenDict.indexSize(), silIdx);
  auto startState = lm->start(false);

  // Insert words
  for (const auto& it : lexicon) {
    const std::string& word = it.first;
    int usrIdx = wordDict.getIndex(word);
    float score = -1;
    LMStatePtr dummyState;
    std::tie(dummyState, score) = lm->score(startState, usrIdx);

    for (const auto& tokens : it.second) {
      auto tokensTensor = tkn2Idx(tokens, tokenDict);
      trie->insert(tokensTensor, usrIdx, score);
    }
  }
  LOG(INFO) << "[Decoder] Trie planted.\n";

  // Smearing
  trie->smear(SmearingMode::MAX);
  LOG(INFO) << "[Decoder] Trie smeared.\n";

  std::vector<float> trieScoreTarget{
      -1.05971, -2.87742, -2.64553, -3.05081, -1.05971, -3.08968};
  for (int i = 0; i < sentence.size(); i++) {
    auto word = sentence[i];
    auto wordTensor = tokens2Tensor(word, tokenDict);
    auto node = trie->search(wordTensor);
    ASSERT_NEAR(node->maxScore, trieScoreTarget[i], 1e-5);
  }

  /* -------- Build Decoder --------*/
  DecoderOptions decoderOpt(
      2500, // FLAGS_beamsize
      100.0, // FLAGS_beamthreshold
      2.0, // FLAGS_lmweight
      2.0, // FLAGS_lexiconcore
      -std::numeric_limits<float>::infinity(), // FLAGS_unkweight
      false, // FLAGS_logadd
      -1, // FLAGS_silweight
      CriterionType::ASG);

  WordLMDecoder decoder(
      decoderOpt, trie, lm, silIdx, blankIdx, unkIdx, transitions);
  LOG(INFO) << "[Decoder] Decoder constructed.\n";

  /* -------- Run --------*/
  auto emission = emissionSet.emissions[0];

  std::vector<float> score;
  std::vector<std::vector<int>> wordPredictions;
  std::vector<std::vector<int>> letterPredictions;

  auto timer = fl::TimeMeter();
  timer.resume();
  auto results = decoder.decode(emission.data(), T, N);
  timer.stop();

  int n_hyp = results.size();

  ASSERT_EQ(n_hyp, 1452);

  std::vector<float> hypScoreTarget{
      -278.111, -278.652, -279.275, -279.847, -280.01};
  for (int i = 0; i < 5; i++) {
    ASSERT_NEAR(results[i].score, hypScoreTarget[i], 1e-3);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

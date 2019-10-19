/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include <flashlight/flashlight.h>

#include "criterion/attention/attention.h"
#include "criterion/criterion.h"

using namespace fl;
using namespace w2l;

TEST(Seq2SeqTest, Seq2Seq) {
  int nclass = 40;
  int hiddendim = 256;
  int batchsize = 2;
  int inputsteps = 200;
  int outputsteps = 50;
  int maxoutputlen = 100;
  int nAttnRound = 2;

  std::vector<std::shared_ptr<AttentionBase>> attentions(
      nAttnRound, std::make_shared<ContentAttention>());
  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 1 /* eos token index */,
      maxoutputlen,
      attentions,
      nullptr,
      false,
      100,
      0.0,
      false,
      w2l::kRandSampling,
      1.0,
      2, // nRnnLayer
      nAttnRound,
      0.0);

  auto input = af::randn(hiddendim, inputsteps, batchsize, f32);
  auto target = af::randu(outputsteps, batchsize, f32) * 0.99 * nclass;
  target = target.as(s32);

  Variable output, attention;
  std::tie(output, attention) =
      seq2seq.vectorizedDecoder(noGrad(input), noGrad(target));

  ASSERT_EQ(output.dims(0), nclass);
  ASSERT_EQ(output.dims(1), outputsteps);
  ASSERT_EQ(output.dims(2), batchsize);
  ASSERT_EQ(output.dims(3), 1);

  ASSERT_EQ(attention.dims(0), outputsteps);
  ASSERT_EQ(attention.dims(1), inputsteps);
  ASSERT_EQ(attention.dims(2), batchsize);

  auto losses = seq2seq({noGrad(input), noGrad(target)}).front();
  ASSERT_EQ(losses.dims(0), batchsize);

  // Backward runs.
  losses.backward();

  // Check that vecotrized decoder and sequential decoder give the same
  // results.
  Variable out_seq, attention_seq;
  std::tie(out_seq, attention_seq) =
      seq2seq.decoder(noGrad(input), noGrad(target));

  ASSERT_TRUE(allClose(output, out_seq, 1e-6));
  ASSERT_TRUE(allClose(attention, attention_seq, 1e-6));

  // Check size 1 Target works
  target = target(0, af::span);
  auto loss = seq2seq({noGrad(input), noGrad(target)}).front();

  // Make sure eval mode is not storing variables.
  seq2seq.eval();
  std::tie(out_seq, attention_seq) =
      seq2seq.decoder(noGrad(input), noGrad(target));
  ASSERT_FALSE(out_seq.isCalcGrad());
  ASSERT_FALSE(attention_seq.isCalcGrad());
}

TEST(Seq2SeqTest, Seq2SeqViterbi) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  af::setSeed(1);
  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 1 /* eos token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()});

  seq2seq.eval();
  auto input = af::randn(hiddendim, inputsteps, 1, f32);

  auto path = seq2seq.viterbiPath(input);
  ASSERT_GT(path.elements(), 0);
  ASSERT_LE(path.elements(), maxoutputlen);
}

TEST(Seq2SeqTest, Seq2SeqBeamSearchViterbi) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 1 /* eos token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()});

  seq2seq.eval();
  auto input = af::randn(hiddendim, inputsteps, 1, f32);

  auto viterbipath = seq2seq.viterbiPath(input);
  auto beampath = seq2seq.beamPath(input, 1);
  ASSERT_EQ(beampath.size(), viterbipath.elements());
  for (int idx = 0; idx < beampath.size(); idx++) {
    ASSERT_EQ(beampath[idx], viterbipath(idx).scalar<int>());
  }
}

TEST(Seq2SeqTest, Seq2SeqMedianWindow) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 1 /* eos token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<MedianWindow>(10, 10));

  seq2seq.eval();
  auto input = af::randn(hiddendim, inputsteps, 1, f32);

  auto viterbipath = seq2seq.viterbiPath(input);
  auto beampath = seq2seq.beamPath(input, 1);
  ASSERT_EQ(beampath.size(), viterbipath.elements());
  for (int idx = 0; idx < beampath.size(); idx++) {
    ASSERT_EQ(beampath[idx], viterbipath(idx).scalar<int>());
  }
}

TEST(Seq2SeqTest, Seq2SeqStepWindow) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 1 /* eos token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<StepWindow>(1, 20, 2.2, 5.8));

  seq2seq.eval();
  auto input = af::randn(hiddendim, inputsteps, 1, f32);

  auto viterbipath = seq2seq.viterbiPath(input);
  auto beampath = seq2seq.beamPath(input, 1);
  ASSERT_EQ(beampath.size(), viterbipath.elements());
  for (int idx = 0; idx < beampath.size(); idx++) {
    ASSERT_EQ(beampath[idx], viterbipath(idx).scalar<int>());
  }
}

TEST(Seq2SeqTest, Seq2SeqStepWindowVectorized) {
  int nclass = 20;
  int hiddendim = 16;
  int batchsize = 2;
  int inputsteps = 20;
  int outputsteps = 10;
  int maxoutputlen = 20;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 1 /* eos token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<StepWindow>(0, 5, 2.2, 5.8),
      true);

  auto input = af::randn(hiddendim, inputsteps, batchsize, f32);
  auto target = af::randu(outputsteps, batchsize, f32) * 0.99 * nclass;
  target = target.as(s32);

  Variable output_v, attention_v, output_s, attention_s;
  std::tie(output_v, attention_v) =
      seq2seq.vectorizedDecoder(noGrad(input), noGrad(target));

  std::tie(output_s, attention_s) =
      seq2seq.decoder(noGrad(input), noGrad(target));

  ASSERT_TRUE(allClose(output_v, output_s, 1e-6));
  ASSERT_TRUE(allClose(attention_v, attention_s, 1e-6));
}

TEST(Seq2SeqTest, Seq2SeqAttn) {
  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100;
  Seq2SeqCriterion seq2seq(
      N,
      H,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<MedianWindow>(2, 3));
  seq2seq.eval();

  auto input = noGrad(af::randn(H, T, B, f32));
  auto target = noGrad((af::randu(U, B, f32) * 0.99 * N).as(s32));

  Variable output, attention;
  std::tie(output, attention) = seq2seq.decoder(input, target);
  // check padding works
  ASSERT_EQ(attention.dims(), af::dim4({U, T, B}));
}

TEST(Seq2SeqTest, Seq2SeqMixedAttn) {
  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100, nHead = 2;
  Seq2SeqCriterion seq2seq(
      N,
      H,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>(),
       std::make_shared<MultiHeadContentAttention>(H, nHead)},
      std::make_shared<StepWindow>(1, 20, 2.2, 5.8),
      false,
      100,
      0.0,
      false,
      w2l::kRandSampling,
      1.0,
      1,
      2);
  seq2seq.eval();

  auto input = noGrad(af::randn(H, T, B, f32));
  auto target = noGrad((af::randu(U, B, f32) * 0.99 * N).as(s32));

  Variable output, attention;
  std::tie(output, attention) = seq2seq.decoder(input, target);
  ASSERT_EQ(attention.dims(), af::dim4({U * nHead, T, B}));
}

TEST(Seq2SeqTest, Serialization) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = "/tmp/" + userstr + "_test.mdl";

  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100, nAttnRound = 2;

  std::vector<std::shared_ptr<AttentionBase>> attentions(
      nAttnRound, std::make_shared<ContentAttention>());

  auto seq2seq = std::make_shared<Seq2SeqCriterion>(
      N,
      H,
      N - 1,
      maxoutputlen,
      attentions,
      std::make_shared<MedianWindow>(2, 3),
      false,
      100,
      0.0,
      false,
      w2l::kRandSampling,
      1.0,
      2, // nRnnLayer
      nAttnRound,
      0.0);
  seq2seq->eval();

  auto input = noGrad(af::randn(H, T, B, f32));
  auto target = noGrad((af::randu(U, B, f32) * 0.99 * N).as(s32));

  Variable output, attention;
  std::tie(output, attention) = seq2seq->decoder(input, target);

  save(path, seq2seq);

  std::shared_ptr<Seq2SeqCriterion> loaded;
  load(path, loaded);
  loaded->eval();

  Variable outputl, attentionl;
  std::tie(outputl, attentionl) = loaded->decoder(input, target);

  ASSERT_TRUE(allParamsClose(*loaded, *seq2seq));
  ASSERT_TRUE(allClose(outputl, output));
  ASSERT_TRUE(allClose(attentionl, attention));
}

TEST(Seq2SeqTest, BatchedDecoderStep) {
  int N = 5, H = 8, B = 10, T = 20, maxoutputlen = 100;
  int nRnnLayer = 2, nAttnRound = 2;
  std::vector<std::shared_ptr<AttentionBase>> contentAttentions(
      nAttnRound, std::make_shared<ContentAttention>());
  std::vector<std::shared_ptr<AttentionBase>> neuralContentAttentions(
      nAttnRound, std::make_shared<NeuralContentAttention>(H));

  std::vector<Seq2SeqCriterion> criterions{Seq2SeqCriterion(
                                               N,
                                               H,
                                               N - 1,
                                               maxoutputlen,
                                               contentAttentions,
                                               nullptr,
                                               false,
                                               100,
                                               0.0,
                                               false,
                                               w2l::kRandSampling,
                                               1.0,
                                               nRnnLayer,
                                               nAttnRound,
                                               0.0),
                                           Seq2SeqCriterion(
                                               N,
                                               H,
                                               N - 1,
                                               maxoutputlen,
                                               neuralContentAttentions,
                                               nullptr,
                                               false,
                                               100,
                                               0.0,
                                               false,
                                               w2l::kRandSampling,
                                               1.0,
                                               nRnnLayer,
                                               nAttnRound,
                                               0.0)};

  for (auto& seq2seq : criterions) {
    seq2seq.eval();
    std::vector<Variable> ys;
    std::vector<Seq2SeqState> inStates(B, Seq2SeqState(nAttnRound));
    std::vector<Seq2SeqState*> inStatePtrs(B);

    auto input = noGrad(af::randn(H, T, 1, f32));
    std::vector<std::vector<float>> single_scores(B);
    std::vector<std::vector<float>> batched_scores;

    for (int i = 0; i < B; i++) {
      Variable y = constant(i % N, 1, s32, false);
      ys.push_back(y);

      inStates[i].alpha = noGrad(af::randn(1, T, 1, f32));
      for (int j = 0; j < nAttnRound; j++) {
        inStates[i].hidden[j] = noGrad(af::randn(H, 1, nRnnLayer, f32));
      }
      inStates[i].summary = noGrad(af::randn(H, 1, 1, f32));
      inStatePtrs[i] = &inStates[i];

      // Single forward
      Seq2SeqState outstate(nAttnRound);
      Variable ox;
      std::tie(ox, outstate) = seq2seq.decodeStep(input, y, inStates[i]);
      ox = logSoftmax(ox, 0);
      single_scores[i] = w2l::afToVector<float>(ox);
    }

    // Batched forward
    std::vector<Seq2SeqStatePtr> outstates;
    std::tie(batched_scores, outstates) =
        seq2seq.decodeBatchStep(input, ys, inStatePtrs);

    // Check
    for (int i = 0; i < B; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(single_scores[i][j], batched_scores[i][j], 1e-5);
      }
    }
  }
}

TEST(Seq2SeqTest, Seq2SeqSampling) {
  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100;
  auto input = noGrad(af::randn(H, T, B, f32));
  auto target = noGrad((af::randu(U, B, f32) * 0.99 * N).as(s32));

  std::vector<std::string> samplingStrategy(
      {w2l::kRandSampling, w2l::kModelSampling});

  for (const auto& ss : samplingStrategy) {
    Seq2SeqCriterion seq2seq(
        N,
        H,
        N - 1,
        maxoutputlen,
        {std::make_shared<ContentAttention>()},
        nullptr,
        false,
        0,
        0.05,
        false,
        ss);
    seq2seq.train();

    Variable output, attention;
    std::tie(output, attention) = seq2seq.decoder(input, target);
    ASSERT_EQ(attention.dims(), af::dim4({U, T, B}));
    ASSERT_EQ(output.dims(), af::dim4({N, U, B, 1}));
  }

  Seq2SeqCriterion seq2seq1(
      N,
      H,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      nullptr,
      false,
      60,
      0.05,
      false,
      w2l::kRandSampling);
  seq2seq1.train();

  Variable output, attention;
  std::tie(output, attention) = seq2seq1.vectorizedDecoder(input, target);
  ASSERT_EQ(attention.dims(), af::dim4({U, T, B}));
  ASSERT_EQ(output.dims(), af::dim4({N, U, B, 1}));

  Seq2SeqCriterion seq2seq2(
      N,
      H,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      nullptr,
      false,
      60,
      0.05,
      false,
      w2l::kModelSampling);
  seq2seq2.train();
  ASSERT_THROW(seq2seq2.vectorizedDecoder(input, target), std::logic_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

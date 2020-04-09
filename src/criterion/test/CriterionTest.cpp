/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include <array>

#include "criterion/criterion.h"

using namespace fl;
using namespace w2l;

namespace {

constexpr float kEpsilon = 1E-5;

void checkZero(const af::array& val, float precision = kEpsilon) {
  ASSERT_LE(af::max<float>(af::abs(val)), precision);
}

using JacobianFunc = std::function<Variable(Variable&)>;
void jacobian_test(
    JacobianFunc func,
    Variable& input,
    float precision = 1E-3,
    float perturbation = 1E-2) {
  auto fwd_jacobian =
      af::array(func(input).elements(), input.elements(), af::dtype::f64);
  for (int i = 0; i < input.elements(); ++i) {
    af::array orig = input.array()(i);
    input.array()(i) = orig - perturbation;
    auto outa = func(input).array();

    input.array()(i) = orig + perturbation;
    auto outb = func(input).array();
    input.array()(i) = orig;
    fwd_jacobian(af::span, i) =
        af::moddims((outb - outa), outa.elements()) * 0.5 / perturbation;
  }

  auto bwd_jacobian =
      af::array(func(input).elements(), input.elements(), af::dtype::f64);
  auto dout =
      Variable(af::constant(0, func(input).dims(), input.type()), false);
  for (int i = 0; i < dout.elements(); ++i) {
    dout.array()(i) = 1; // element in 1D view
    input.zeroGrad();
    auto out = func(input);
    out.backward(dout);
    bwd_jacobian(i, af::span) =
        af::moddims(input.grad().array(), input.elements());
    dout.array()(i) = 0;
  }

  checkZero(fwd_jacobian - bwd_jacobian, precision);
}

} // namespace

TEST(CriterionTest, CTCEmptyTarget) {
  // Non-empty input, Empty target, batchsize > 0
  auto input = Variable(af::array(3, 2, 5), true);
  auto target = Variable(af::array(0, 5), false);
  auto ctc = ConnectionistTemporalClassificationCriterion();
  auto loss = ctc({input, target}).front();
  loss.backward();
  ASSERT_FALSE(af::anyTrue<bool>(af::isNaN(loss.array())));

  auto func_conv_in = [&](Variable& inp) {
    return ctc.forward({inp, target}).front();
  };
  jacobian_test(func_conv_in, input);
}

TEST(CriterionTest, CTCCost) {
  // Test case: 1
  auto neginf = -std::numeric_limits<float>::infinity();
  std::array<float, 6> input1 = {0.0, neginf, neginf, 0.0, 0.0, neginf};
  std::array<int, 2> target1 = {0, 0};
  const int N1 = 2, L1 = 2, T1 = 3;

  auto ctc1 = ConnectionistTemporalClassificationCriterion();
  auto input1af = Variable(af::array(N1, T1, input1.data()), true);
  auto target1af = Variable(af::array(L1, target1.data()), false);

  auto loss1 = ctc1({input1af, target1af}).front();
  ASSERT_NEAR(loss1.scalar<float>(), 0.0, kEpsilon);

  // Test case: 2
  std::array<int, 2> target2 = {1, 2};
  int N2 = 4, L2 = 2, T2 = 3;

  auto ctc2 = ConnectionistTemporalClassificationCriterion();
  auto input2af = Variable(af::constant(0.0, N2, T2, f32), true);
  auto target2af = Variable(af::array(L2, target2.data()), false);

  auto loss2 = ctc2({input2af, target2af}).front();
  ASSERT_NEAR(loss2.scalar<float>(), -log(0.25 * 0.25 * 0.25 * 5), kEpsilon);
}

TEST(CriterionTest, CTCJacobian) {
  int N = 30, T = 80, L = 20;
  auto in = Variable(af::log(af::randu(N, T)), true);
  auto t = af::abs(af::randu(L, af::dtype::s32)) % (N - 2);
  auto tgt = Variable(t.as(af::dtype::s32), false);
  auto l = ConnectionistTemporalClassificationCriterion(
      w2l::CriterionScaleMode::INPUT_SZ_SQRT);
  auto func_conv_in = [&](Variable& inp) {
    return l.forward({inp, tgt}).front();
  };
  jacobian_test(func_conv_in, in);
}

TEST(CriterionTest, Batching) {
  {
    int N = 10, T = 25, L = 15, B = 5;
    auto in = Variable(af::log(af::randu(N, T, B)), true);
    auto t = af::abs(af::randu(L, B, af::dtype::s32)) % (N - 2);
    for (int i = 0; i < B; ++i) {
      int r = rand() % L;
      if (r > 0) {
        t(af::seq(r, af::end), i) = -1;
      }
    }
    auto tgt = Variable(t.as(af::dtype::s32), false);
    auto l = ConnectionistTemporalClassificationCriterion(
        w2l::CriterionScaleMode::TARGET_SZ_SQRT);
    auto func_conv_in = [&](Variable& inp) {
      return l.forward({inp, tgt}).front();
    };
    jacobian_test(func_conv_in, in);
  }
  {
    int N = 80, T = 50, L = 25, B = 10;
    auto in = Variable(af::log(af::randu(N, T, B)), true);
    auto t = af::abs(af::randu(L, B, af::dtype::s32)) % (N - 2);
    for (int i = 0; i < B; ++i) {
      int r = rand() % L;
      if (r > 0) {
        t(af::seq(r, af::end), i) = -1;
      }
    }
    auto tgt = Variable(t.as(af::dtype::s32), false);
    auto l = ConnectionistTemporalClassificationCriterion(
        w2l::CriterionScaleMode::TARGET_SZ);
    auto output = l.forward({in, tgt}).front();

    for (int i = 0; i < B; ++i) {
      auto output_i = l.forward({in.slice(i), tgt.col(i)}).front();
      checkZero(output.array()(i) - output_i.array(), 1E-6);
    }
  }
}

TEST(CriterionTest, CTCCompareTensorflow) {
  // The following test cases are taken from Tensor Flow CTC implementation
  // tinyurl.com/y9du5v5a

  // Test Case: 1
  const int T1 = 5, N1 = 6, L1 = 5;
  std::array<int, 5> target1 = {0, 1, 2, 1, 0};
  float loss_expected1 = 3.34211;
  std::array<float, N1* T1> input1 = {
      0.633766,  0.221185, 0.0917319, 0.0129757,  0.0142857,  0.0260553,
      0.111121,  0.588392, 0.278779,  0.0055756,  0.00569609, 0.010436,
      0.0357786, 0.633813, 0.321418,  0.00249248, 0.00272882, 0.0037688,
      0.0663296, 0.643849, 0.280111,  0.00283995, 0.0035545,  0.00331533,
      0.458235,  0.396634, 0.123377,  0.00648837, 0.00903441, 0.00623107,
  };
  std::transform(
      input1.begin(), input1.end(), input1.begin(), [](float p) -> float {
        return log(p);
      });
  std::array<float, N1* T1> grad_expected1 = {
      -0.366234, 0.221185,  0.0917319, 0.0129757,  0.0142857,  0.0260553,
      0.111121,  -0.411608, 0.278779,  0.0055756,  0.00569609, 0.010436,
      0.0357786, 0.633813,  -0.678582, 0.00249248, 0.00272882, 0.0037688,
      0.0663296, -0.356151, 0.280111,  0.00283995, 0.0035545,  0.00331533,
      -0.541765, 0.396634,  0.123377,  0.00648837, 0.00903441, 0.00623107};

  auto ctc1 = ConnectionistTemporalClassificationCriterion();
  auto input1af = Variable(af::array(N1, T1, input1.data()), true);
  auto target1af = Variable(af::array(L1, target1.data()), false);
  auto grad_expected1af =
      Variable(af::array(N1, T1, grad_expected1.data()), false);

  auto loss1 = ctc1({input1af, target1af}).front();
  ASSERT_NEAR(loss1.scalar<float>(), loss_expected1, kEpsilon);

  loss1.backward();
  checkZero(input1af.grad().array() - grad_expected1af.array());

  // Test Case: 2
  const int T2 = 5, N2 = 6, L2 = 4;
  std::array<int, 4> target2 = {0, 1, 1, 0};
  float loss_expected2 = 5.42262;
  std::array<float, N2* T2> input2 = {
      0.30176,  0.28562,  0.0831517, 0.0862751, 0.0816851, 0.161508,
      0.24082,  0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549,
      0.230246, 0.450868, 0.0389607, 0.038309,  0.0391602, 0.202456,
      0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345,
      0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046,
  };
  std::transform(
      input2.begin(), input2.end(), input2.begin(), [](float p) -> float {
        return log(p);
      });
  std::array<float, N2* T2> grad_expected2 = {
      -0.69824,  0.28562,   0.0831517, 0.0862751, 0.0816851, 0.161508,
      0.24082,   -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549,
      0.230246,  0.450868,  0.0389607, 0.038309,  0.0391602, -0.797544,
      0.280884,  -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345,
      -0.576714, 0.315517,  0.0338439, 0.0393744, 0.0339315, 0.154046,
  };

  auto ctc2 = ConnectionistTemporalClassificationCriterion();
  auto input2af = Variable(af::array(N2, T2, input2.data()), true);
  auto target2af = Variable(af::array(L2, target2.data()), false);
  auto grad_expected2af =
      Variable(af::array(N2, T2, grad_expected2.data()), false);

  auto loss2 = ctc2({input2af, target2af}).front();
  ASSERT_NEAR(loss2.scalar<float>(), loss_expected2, kEpsilon);

  loss2.backward();
  checkZero(input2af.grad().array() - grad_expected2af.array());
}

TEST(CriterionTest, ViterbiPath) {
  // Test case: 1
  auto in = af::randu(4, 5); // All values < 1
  std::array<int, 5> expectedpath1 = {3, 2, 0, 2, 2};
  for (int j = 0; j < 5; ++j) {
    in(expectedpath1[j], j) = 2;
  }
  ConnectionistTemporalClassificationCriterion ctc;
  auto vpath1Arr = ctc.viterbiPath(in);
  af::array expPath1Arr = af::array(5, expectedpath1.data());
  checkZero(vpath1Arr - expPath1Arr);

  // test batch input
  auto intile = af::tile(in, 1, 1, 2);
  auto vpath1bArr = ctc.viterbiPath(intile);
  checkZero(vpath1bArr - af::tile(expPath1Arr, 1, 2));

  // Test case: 2
  constexpr int T2 = 4, N2 = 3;

  // clang-format off
  std::array<float, (T2 * N2)> input2Vec = {
    0, 0, 7,
    5, 4, 3,
    5, 8, 5,
    5, 4, 3,
  };
  std::array<float, (N2 * N2)> trans2Vec = {
    0, 2, 0,
    0, 0, 2,
    2, 0, 0,
  };
  std::array<int, T2> expectedPath2Vec = {2, 1, 1, 0};
  // clang-format on

  af::array input2(N2, T2, input2Vec.data());
  af::array trans2(N2, N2, trans2Vec.data());
  af::array expectedPath2(T2, expectedPath2Vec.data());

  AutoSegmentationCriterion asg(N2);
  asg.setParams(Variable(trans2, true), 0);
  auto path2 = asg.viterbiPath(input2);
  checkZero(path2 - expectedPath2);

  // Test case: 2b (batching)
  auto input2b = af::tile(input2, 1, 1, 77);
  auto expectedPath2b = af::tile(expectedPath2, 1, 77);
  auto path2b = asg.viterbiPath(input2b);
  checkZero(path2b - expectedPath2b);

  // If trasition probablities are same, CTC and ASG viterbi paths should match
  AutoSegmentationCriterion asg2(30);
  asg.param(0).array() = af::constant(1.0, 30, 30);
  for (int t = 1; t < 5; ++t) {
    af::array randInput = af::randu(30, t * 10, t);
    auto asgPathArr = asg2.viterbiPath(randInput);
    auto ctcPathArr = ctc.viterbiPath(randInput);
    ASSERT_EQ(asgPathArr.dims(), asgPathArr.dims());
    checkZero(asgPathArr - ctcPathArr);
  }
}

// Test with alternating blanks and with varying target sizes
TEST(CriterionTest, ASGAlternatingBlanks) {
  w2l::AutoSegmentationCriterion criterion(2);
  int C = 2; // (one class + blank)
  int T = 7;
  int mL = 3;
  int B = 2;
  std::vector<float> x_v = {
      -0x1.1f60fap+1, -0x1.0518e2p+0, 0x1.2016e2p-3,  -0x1.dfe0dp-4,
      0x1.00ee32p-2,  0x1.af74fp-2,   -0x1.f29964p-2, 0x1.977e08p-2,
      -0x1.52548ep-1, -0x1.ae9504p-3, 0x1.bcf1fcp+1,  0x1.31ad5p+0,
      0x1.9bc5aep-1,  0x1.3c7dacp-1,  0x1.3e2852p-1,  0x1.6699f4p-1,
      0x1.095a5p+0,   0x1.1840bcp-1,  0x1.465a4ep-1,  0x1.2c4cacp-1,
      0x1.754998p-1,  0x1.cb6698p-2,  -0x1.1cadcp+0,  0x1.757b88p-2,
      0x1.3dec32p+0,  0x1.320fp+0,    -0x1.9eb1a4p-1, -0x1.e43beap-2};
  af::array x = af::array(af::dim4(C, T, B), x_v.data());
  af::array y = af::constant(-1, af::dim4(mL * 2 + 1, B), s32);
  int L;
  L = 2;
  y(af::seq(0, 2 * L, 2), 0) = 1;
  y(af::seq(1, 2 * L - 1, 2), 0) = 0;
  L = 3;
  y(af::seq(0, 2 * L, 2), 1) = 1;
  y(af::seq(1, 2 * L - 1, 2), 1) = 0;
  af::array expectedPath = af::constant(1, af::dim4(T, B));
  expectedPath(1, 0) = 0;
  expectedPath(5, 0) = 0;
  expectedPath(1, 1) = 0;
  expectedPath(3, 1) = 0;
  expectedPath(5, 1) = 0;
  af::array path = criterion.viterbiPath(x, y);
  checkZero(path - expectedPath);
}

// Test constrained viterbi path for ctc and asg criterion.
// Target will be a range from [0, 1, 2, 3]
// Input will predict a high probability for each target i at frame i * 2 + 1
// Expected output is [0, 0, 1, 1, 2, 2, 3, 3] for both ctc and ASG with
// constant transitions
TEST(CriterionTest, VertibitPathConstrained) {
  const int B = 2;
  const int T = 8;
  const int N = 5;
  const int L = 4;
  af::array target = af::range(af::dim4(L, B), 0, s32);
  af::array input = af::constant(0.01, N, T, B);
  af::array expectedPath = af::constant(0, T, B, s32);
  for (int i = 0; i < L; i++) {
    input(i, i * 2 + 1, af::span) = 1.0;
    expectedPath(i * 2, af::span) = i;
    expectedPath(i * 2 + 1, af::span) = i;
  }

  ConnectionistTemporalClassificationCriterion ctc;
  af::array ctcPath = ctc.viterbiPath(input, target);
  af::array diff = ctcPath - expectedPath;
  ASSERT_LE(af::max<float>(af::abs(diff)), kEpsilon);

  AutoSegmentationCriterion asg(N);
  asg.param(0).array() = af::constant(1.0, N, N);
  af::array asgPath = asg.viterbiPath(input, target);
  diff = asgPath - expectedPath;
  ASSERT_LE(af::max<float>(af::abs(diff)), kEpsilon);
}

// Test that CTC can return a path with no spaces
TEST(CriterionTest, CTCViterbiPathNopaces) {
  const int B = 3; // Batchsize
  const int T = 10; // Utterance length
  const int N = 30; // Number of tokens
  const int L = 1; // Length of target
  const int target_idx = 1; // Token Idx of target

  af::array input = af::constant(0.01, N, T, B);
  af::array target = af::constant(0, L, B, s32);
  af::array expectedPath = af::constant(target_idx, T, B, s32);

  // Target_idx has the highest prob for all t in T
  input(target_idx, af::span, af::span) = 1.0;
  target(af::span, af::span) = target_idx;

  ConnectionistTemporalClassificationCriterion ctc;
  af::array vpathArr = ctc.viterbiPath(input, target);
  af::array diff = expectedPath - vpathArr;
  ASSERT_LE(af::max<float>(af::abs(diff)), kEpsilon);
}

// Test that CTC can return a path that optionally ends with a space
TEST(CriterionTest, CTCViterbiPathConstrainedEndWithSpace) {
  const int B = 3; // Batchsize
  const int T = 10; // Utterance length
  const int N = 30; // Number of tokens
  const int blank_label = N - 1;
  const int L = 1; // Length of target
  const int target_idx = 1; // Token Idx of target

  af::array input = af::constant(0.01, N, T, B);
  af::array target = af::constant(target_idx, L, B, s32);
  af::array expectedPath = af::constant(target_idx, T, B, s32);
  // Target_idx has the highest prob for all t in T, except for T - 1, which is
  // a blank label
  input(target_idx, af::span, af::span) = 1.0;
  input(target_idx, T - 1, af::span) = 0.00;
  input(blank_label, T - 1, af::span) = 1.0;
  expectedPath(T - 1, af::span) = blank_label;

  ConnectionistTemporalClassificationCriterion ctc;
  af::array vpathArr = ctc.viterbiPath(input, target);
  af::array diff = expectedPath - vpathArr;
  ASSERT_LE(af::max<float>(af::abs(diff)), kEpsilon);
}

// Test that CTC can return a path that optionally begins with a space
TEST(CriterionTest, CTCViterbiPathConstrainedBeginWithSpace) {
  const int B = 3; // Batchsize
  const int T = 10; // Utterance length
  const int N = 30; // Number of tokens
  const int blank_label = N - 1;
  const int L = 1; // Length of target
  const int target_idx = 1; // Token Idx of target

  // Target_idx has the highest prob for all t in T, except for 0, which is
  // a blank label
  af::array input = af::constant(0.01, N, T, B);
  af::array target = af::constant(target_idx, L, B, s32);
  af::array expectedPath = af::constant(target_idx, T, B, s32);
  input(target_idx, af::span, af::span) = 1.0;
  input(target_idx, 0, af::span) = 0.00;
  input(blank_label, 0, af::span) = 1.0;
  expectedPath(0, af::span) = blank_label;

  ConnectionistTemporalClassificationCriterion ctc;
  af::array vpathArr = ctc.viterbiPath(input, target);
  af::array diff = expectedPath - vpathArr;
  ASSERT_LE(af::max<float>(af::abs(diff)), kEpsilon);
}

// Test that CTC can return a path that optionally begins and ends with a space
TEST(CriterionTest, CTCViterbiPathConstrainedBeginAndEndWithSpace) {
  const int B = 3; // Batchsize
  const int T = 10; // Utterance length
  const int N = 30; // Number of tokens
  const int blank_label = N - 1;
  const int L = 1; // Length of target
  const int target_idx = 1; // Token Idx of target

  // Target_idx has the highest prob for all t in T, except for 0 and T -1
  // which is // a blank label
  af::array input = af::constant(0.01, N, T, B);
  af::array target = af::constant(target_idx, L, B, s32);
  af::array expectedPath = af::constant(target_idx, T, B, s32);
  input(target_idx, af::span, af::span) = 1.0;
  input(target_idx, 0, af::span) = 0.00;
  input(blank_label, 0, af::span) = 1.0;
  expectedPath(0, af::span) = blank_label;

  input(target_idx, T - 1, af::span) = 0.00;
  expectedPath(T - 1, af::span) = blank_label;

  ConnectionistTemporalClassificationCriterion ctc;
  af::array vpathArr = ctc.viterbiPath(input, target);
  af::array diff = expectedPath - vpathArr;
  ASSERT_LE(af::max<float>(af::abs(diff)), kEpsilon);
}

TEST(CriterionTest, FCCCost) {
  // Test case: 1
  std::array<float, 12> input1 = {
      1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  std::transform(
      input1.begin(), input1.end(), input1.begin(), [](float p) -> float {
        return log(p);
      });
  std::array<int, 2> dummy_target1 = {0, 0};
  const int N1 = 2, L1 = 2, T1 = 3, B1 = 2;

  auto fcc1 = FullConnectionCriterion(N1);
  auto input1af = Variable(af::array(N1, T1, B1, input1.data()), true);
  auto target1af = Variable(af::array(L1, dummy_target1.data()), false);

  auto l1 = fcc1(input1af, target1af);
  std::vector<float> loss1_host(B1);
  l1.host(loss1_host.data());
  ASSERT_NEAR(loss1_host[0], 0.0, kEpsilon);
  ASSERT_NEAR(loss1_host[1], 0.0, kEpsilon);

  // Test case: 2
  std::array<float, 12> input2;
  std::fill_n(input2.data(), 12, log(0.25));
  std::array<int, 2> dummy_target2 = {1, 2};
  int N2 = 4, L2 = 2, T2 = 3;

  auto fcc2 = FullConnectionCriterion(N2);
  auto input2af = Variable(af::array(N2, T2, input2.data()), true);
  auto target2af = Variable(af::array(L2, dummy_target2.data()), false);

  auto l2 = fcc2(input2af, target2af);
  std::vector<float> loss2_host(1);
  l2.host(loss2_host.data());
  ASSERT_NEAR(loss2_host[0], 0.0, kEpsilon);

  // Test case: 3
  int N3 = 40, T3 = 300, L3 = 50, B3 = 3;
  auto in = logSoftmax(Variable(af::randu(N3, T3, B3), true), 0);
  auto t = af::abs(af::randu(L3, B3, af::dtype::s32)) % (N3 - 1);
  auto tgt = Variable(t.as(af::dtype::s32), false);
  auto fcc3 = FullConnectionCriterion(N3);

  auto l3 = fcc3(in, tgt);
  std::vector<float> loss3_host(B3);
  l3.host(loss3_host.data());
  ASSERT_NEAR(loss3_host[0], 0.0, kEpsilon);
  ASSERT_NEAR(loss3_host[1], 0.0, kEpsilon);
  ASSERT_NEAR(loss3_host[2], 0.0, kEpsilon);
}

TEST(CriterionTest, FCCJacobian) {
  int N = 3, T = 8, L = 1, B = 2;
  auto in = Variable(af::log(af::randu(N, T, B)), true);
  auto t = af::abs(af::randu(L, B, af::dtype::s32)) % (N - 1);
  auto tgt = Variable(t.as(af::dtype::s32), false);
  auto l = FullConnectionCriterion(N, w2l::CriterionScaleMode::TARGET_SZ_SQRT);

  // Test case for input
  auto func_in = [&](Variable& inp) { return l.forward(inp, tgt); };
  jacobian_test(func_in, in);

  // Test case for transition
  auto transition = Variable(af::randu(N, N), true);
  auto func_trans = [&](Variable& transition_p) {
    l.setParams(transition_p, 0);
    return l.forward(in, tgt);
  };
  jacobian_test(func_trans, transition);
}

TEST(CriterionTest, FACCost) {
  // Test case: 1
  std::array<float, 12> input1 = {
      1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  std::array<int, 4> target1 = {0, 1, 0, 1};
  const int N1 = 2, L1 = 2, T1 = 3, B1 = 2;

  auto fac1 = ForceAlignmentCriterion(N1);
  auto input1af = Variable(af::array(N1, T1, B1, input1.data()), true);
  auto target1af = Variable(af::array(L1, B1, target1.data()), false);

  auto loss1 = fac1(input1af, target1af);
  std::vector<float> loss1_host(B1);
  loss1.host(loss1_host.data());
  ASSERT_NEAR(loss1_host[0], log(exp(1.5) + exp(2.5)), kEpsilon);
  ASSERT_NEAR(loss1_host[1], log(exp(2) + exp(3)), kEpsilon);

  // Test case: 2
  std::array<float, 12> input2;
  std::fill_n(input2.data(), 12, log(0.25));
  std::array<int, 2> target2 = {0, 1};
  int N2 = 4, L2 = 2, T2 = 3;

  auto fac2 = ForceAlignmentCriterion(N2);
  auto input2af = Variable(af::array(N2, T2, input2.data()), true);
  auto target2af = Variable(af::array(L2, target2.data()), false);

  auto loss2 = fac2(input2af, target2af);
  ASSERT_NEAR(loss2.scalar<float>(), -log(32), kEpsilon);
}

TEST(CriterionTest, FACJacobian) {
  int N = 3, T = 10, B = 3, L = 3;
  auto in = Variable(af::log(af::randu(N, T, B)), true);
  std::array<int, 9> target = {0, 1, -1, 1, -1, -1, 0, 2, 1};
  auto tgt = Variable(af::array(L, B, target.data()), false);
  auto l = ForceAlignmentCriterion(N, w2l::CriterionScaleMode::TARGET_SZ_SQRT);

  // Test case for input
  auto func_in = [&](Variable& inp) { return l.forward(inp, tgt); };
  jacobian_test(func_in, in);

  // Test case for transition
  auto transition = Variable(af::randu(N, N), true);
  auto func_trans = [&](Variable& transition_p) {
    l.setParams(transition_p, 0);
    return l.forward(in, tgt);
  };
  jacobian_test(func_trans, transition);
}

TEST(CriterionTest, ASGCost) {
  // Test case: 1
  constexpr int N1 = 2, L1 = 2, T1 = 3, B1 = 2;
  std::array<float, (B1 * T1 * N1)> input1 = {
      1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  std::transform(
      input1.begin(), input1.end(), input1.begin(), [](float p) -> float {
        return log(p);
      });
  std::array<int, (B1 * L1)> target1 = {0, 1, 0, 1};
  std::array<float, (N1 * N1)> trans1 = {};

  auto asg1 = AutoSegmentationCriterion(N1);
  auto input1af = Variable(af::array(N1, T1, B1, input1.data()), true);
  auto target1af = Variable(af::array(L1, B1, target1.data()), false);
  asg1.setParams(Variable(af::array(N1, N1, trans1.data()), true), 0);

  auto loss1 = asg1({input1af, target1af}).front();
  std::vector<float> loss1_host(B1);
  loss1.host(loss1_host.data());
  ASSERT_NEAR(loss1_host[0], -log(0.5), kEpsilon);
  ASSERT_NEAR(loss1_host[1], 0.0, kEpsilon);

  // Test case: 2
  constexpr int N2 = 4, L2 = 2, T2 = 3;
  std::array<float, (N2 * T2)> input2;
  std::fill(input2.begin(), input2.end(), log(0.25));
  std::array<int, L2> target2 = {0, 1};
  std::array<float, (N2 * N2)> trans2 = {};

  auto asg2 = AutoSegmentationCriterion(N2);
  auto input2af = Variable(af::array(N2, T2, input2.data()), true);
  auto target2af = Variable(af::array(L2, target2.data()), false);
  asg2.setParams(Variable(af::array(N2, N2, trans2.data()), true), 0);

  auto loss2 = asg2({input2af, target2af}).front();
  ASSERT_NEAR(loss2.scalar<float>(), log(32), kEpsilon);

  // Test case: 3
  constexpr int N3 = 4, L3_1 = 4, L3_2 = 3, T3 = 3;
  std::array<float, (N3 * T3)> input3;
  std::fill(input3.begin(), input3.end(), log(0.25));
  std::array<int, L3_1> target3 = {0, 1, 1, 1};
  std::array<float, (N3 * N3)> trans3 = {};

  auto asg3 = AutoSegmentationCriterion(N3);
  auto input3af = Variable(af::array(N3, T3, input3.data()), true);
  auto target3af1 = Variable(af::array(L3_1, target3.data()), false);
  auto target3af2 = Variable(af::array(L3_2, target3.data()), false);
  asg3.setParams(Variable(af::array(N3, N3, trans3.data()), true), 0);
  // check if target is truncated
  checkZero(
      asg3({input3af, target3af1}).front().array() -
          asg3({input3af, target3af2}).front().array(),
      1E-5);
}

TEST(CriterionTest, ASGJacobian) {
  int N = 3, T = 10, B = 3, L = 3;
  auto in = Variable(af::log(af::randu(N, T, B)), true);
  std::array<int, 9> target = {0, 1, -1, 1, -1, -1, 0, 2, 1};
  auto tgt = Variable(af::array(L, B, target.data()), false);
  auto l =
      AutoSegmentationCriterion(N, w2l::CriterionScaleMode::TARGET_SZ_SQRT);

  // Test case for input
  auto func_in = [&](Variable& inp) { return l.forward({inp, tgt}).front(); };
  jacobian_test(func_in, in);

  // Test case for transition
  auto transition = Variable(af::randu(N, N), true);
  auto func_trans = [&](Variable& transition_p) {
    l.setParams(transition_p, 0);
    return l.forward({in, tgt}).front();
  };
  jacobian_test(func_trans, transition);
}

TEST(CriterionTest, LinSegJacobian) {
  int N = 3, T = 10, B = 3, L = 3;
  auto in = Variable(af::log(af::randu(N, T, B)), true);
  std::array<int, 9> target = {0, 1, -1, 1, -1, -1, 0, 2, 1};
  auto tgt = Variable(af::array(L, B, target.data()), false);
  auto l =
      LinearSegmentationCriterion(N, w2l::CriterionScaleMode::TARGET_SZ_SQRT);

  // Test case for input
  auto func_in = [&](Variable& inp) { return l.forward({inp, tgt}).front(); };
  jacobian_test(func_in, in);

  // Test case for transition
  auto transition = Variable(af::randu(N, N), true);
  auto func_trans = [&](Variable& transition_p) {
    l.setParams(transition_p, 0);
    return l.forward({in, tgt}).front();
  };
  jacobian_test(func_trans, transition);
}

TEST(CriterionTest, ASGBatching) {
  int N = 80, T = 50, L = 25, B = 10;
  auto in = Variable(af::log(af::randu(N, T, B)), true);
  auto t = af::abs(af::randu(L, B, af::dtype::s32)) % (N - 2);
  for (int i = 0; i < B; ++i) {
    int r = std::rand() % L;
    if (r > 0) {
      t(af::seq(r, af::end), i) = -1;
    }
  }
  auto tgt = Variable(t.as(af::dtype::s32), false);
  auto l = AutoSegmentationCriterion(N, w2l::CriterionScaleMode::TARGET_SZ);
  auto output = l.forward({in, tgt}).front();

  for (int i = 0; i < B; ++i) {
    auto output_i = l.forward({in.slice(i), tgt.col(i)}).front();
    checkZero(output.array()(i) - output_i.array(), 1E-6);
  }
}

TEST(CriterionTest, ASGCompareLua) {
  // Compare with lua version
  const int N = 6, L = 5, T = 5, B = 3;

  // clang-format off
  std::array<float, N * T * B> input = {
    -0.4340, -0.0254,  0.3667,  0.4180, -0.3805, -0.1707,
     0.1060,  0.3631, -0.1122, -0.3825, -0.0031, -0.3801,
     0.0443, -0.3795,  0.3194, -0.3130,  0.0094,  0.1560,
     0.1252,  0.2877,  0.1997, -0.4554,  0.2774, -0.2526,
    -0.4001, -0.2402,  0.1295,  0.0172,  0.1805, -0.3299,

     0.3298, -0.2259, -0.0959,  0.4909,  0.2996, -0.2543,
    -0.2863,  0.3239, -0.3988,  0.0732, -0.2107, -0.4739,
    -0.0906,  0.0480, -0.1301,  0.3975, -0.3317, -0.1967,
     0.4372, -0.2006,  0.0094,  0.3281,  0.1873, -0.2945,
     0.2399,  0.0320, -0.3768, -0.2849, -0.2248,  0.3186,

     0.0225, -0.3867, -0.1929, -0.2904, -0.4958, -0.2533,
     0.4001, -0.1517, -0.2799, -0.2915,  0.4198,  0.4506,
     0.1446, -0.4753, -0.0711,  0.2876, -0.1851, -0.1066,
     0.2081, -0.1190, -0.3902, -0.1668,  0.1911, -0.2848,
    -0.3846,  0.1175,  0.1052,  0.2172, -0.0362,  0.3055,
  };
  std::array<int, L * B> target = {
    2,  1,  5,  1,  3,
    4,  3,  5, -1, -1,
    3,  2,  2,  1, -1,
  };

  std::vector<float> expected_loss = {
    7.7417464256287,
    6.4200420379639,
    8.2780694961548,
  };
  std::array<float, N * T * B> expected_input_grad = {
    0.1060,  0.1595, -0.7639,  0.2485,  0.1118,  0.1380,
    0.1915, -0.7524,  0.1539,  0.1175,  0.1717,  0.1178,
    0.1738,  0.1137,  0.2288,  0.1216,  0.1678, -0.8057,
    0.1766, -0.7923,  0.1902,  0.0988,  0.2056,  0.1210,
    0.1212,  0.1422,  0.2059, -0.8160,  0.2166,  0.1300,

    0.2029,  0.1164,  0.1325,  0.2383, -0.8032,  0.1131,
    0.1414,  0.2602,  0.1263, -0.3441, -0.3009,  0.1172,
    0.1557,  0.1788,  0.1496, -0.5498,  0.0140,  0.0516,
    0.2306,  0.1219,  0.1503, -0.4244,  0.1796, -0.2579,
    0.2149,  0.1745,  0.1160,  0.1271,  0.1350, -0.7675,

    0.2195,  0.1458,  0.1770, -0.8395,  0.1307,  0.1666,
    0.2148,  0.1237, -0.6613, -0.1223,  0.2191,  0.2259,
    0.2002,  0.1077, -0.8386,  0.2310,  0.1440,  0.1557,
    0.2197, -0.1466, -0.5742,  0.1510,  0.2160,  0.1342,
    0.1050, -0.8265,  0.1714,  0.1917,  0.1488,  0.2094,
  };
  std::array<float, N * N> expected_trans_grad = {
    0.3990,  0.3396,  0.3486,  0.3922,  0.3504,  0.3155,
    0.3666,  0.0116, -1.6678,  0.3737,  0.3361, -0.7152,
    0.3468,  0.3163, -1.1583, -0.6803,  0.3216,  0.2722,
    0.3694, -0.6688,  0.3047, -0.8531, -0.6571,  0.2870,
    0.3866,  0.3321,  0.3447,  0.3664, -0.2163,  0.3039,
    0.3640, -0.6943,  0.2988, -0.6722,  0.3215, -0.1860,
  };
  // clang-format on

  auto asg = AutoSegmentationCriterion(N);
  auto input_af = Variable(af::array(N, T, B, input.data()), true);
  auto target_af = Variable(af::array(L, B, target.data()), false);
  asg.setParams(constant(0.0, af::dim4(N, N)), 0);

  auto loss = asg({input_af, target_af}).front();
  std::vector<float> loss_host(B);
  loss.host(loss_host.data());
  for (int i = 0; i < B; i++) {
    ASSERT_NEAR(loss_host[i], expected_loss[i], 1e-3);
  }

  loss.backward();
  auto input_grad = input_af.grad().array();
  checkZero(input_grad - af::array(N, T, B, expected_input_grad.data()), 1e-4);
  auto trans_grad = asg.param(0).grad().array();
  checkZero(trans_grad - af::array(N, N, expected_trans_grad.data()), 1e-4);
}

TEST(CriterionTest, LinSegCompareLua) {
  // Compare LinSegCriterion with lua version
  constexpr int N = 6, L = 5, T = 5, B = 3;

  // clang-format off
  std::array<float, N * T * B> input = {
    -0.4340, -0.0254,  0.3667,  0.4180, -0.3805, -0.1707,
     0.1060,  0.3631, -0.1122, -0.3825, -0.0031, -0.3801,
     0.0443, -0.3795,  0.3194, -0.3130,  0.0094,  0.1560,
     0.1252,  0.2877,  0.1997, -0.4554,  0.2774, -0.2526,
    -0.4001, -0.2402,  0.1295,  0.0172,  0.1805, -0.3299,

     0.3298, -0.2259, -0.0959,  0.4909,  0.2996, -0.2543,
    -0.2863,  0.3239, -0.3988,  0.0732, -0.2107, -0.4739,
    -0.0906,  0.0480, -0.1301,  0.3975, -0.3317, -0.1967,
     0.4372, -0.2006,  0.0094,  0.3281,  0.1873, -0.2945,
     0.2399,  0.0320, -0.3768, -0.2849, -0.2248,  0.3186,

     0.0225, -0.3867, -0.1929, -0.2904, -0.4958, -0.2533,
     0.4001, -0.1517, -0.2799, -0.2915,  0.4198,  0.4506,
     0.1446, -0.4753, -0.0711,  0.2876, -0.1851, -0.1066,
     0.2081, -0.1190, -0.3902, -0.1668,  0.1911, -0.2848,
    -0.3846,  0.1175,  0.1052,  0.2172, -0.0362,  0.3055,
  };
  // target is zero-indexed here; add 1 for Lua counterpart
  std::array<int, L * B> target = {
     2,  1,  5,  1,  3,
     4,  3,  5, -1, -1,
     3,  2,  2,  1, -1,
  };
  // clang-format on

  auto linseg =
      LinearSegmentationCriterion(N, w2l::CriterionScaleMode::TARGET_SZ_SQRT);
  auto input_af = Variable(af::array(N, T, B, input.data()), true);
  auto target_af = Variable(af::array(L, B, target.data()), false);
  linseg.setParams(constant(0.0, af::dim4(N, N)), 0);

  // clang-format off
  std::vector<float> expected_loss = {
    3.4622850827983,
    3.5390825164779,
    4.359541315858,
  };
  std::array<float, N * T * B> expected_input_grad = {
    0.0474,  0.0713, -0.3416,  0.1112,  0.0500,  0.0617,
    0.0856, -0.3365,  0.0688,  0.0525,  0.0768,  0.0527,
    0.0777,  0.0509,  0.1023,  0.0544,  0.0750, -0.3603,
    0.0790, -0.3543,  0.0851,  0.0442,  0.0920,  0.0541,
    0.0542,  0.0636,  0.0921, -0.3649,  0.0969,  0.0582,

    0.0907,  0.0520,  0.0593,  0.1066, -0.3592,  0.0506,
    0.0632,  0.1164,  0.0565,  0.0906, -0.3790,  0.0524,
    0.0696,  0.0800,  0.0669, -0.3338,  0.0547,  0.0626,
    0.1031,  0.0545,  0.0672, -0.3548,  0.0803,  0.0496,
    0.0961,  0.0781,  0.0519,  0.0569,  0.0604, -0.3433,

    0.0982,  0.0652,  0.0791, -0.3754,  0.0585,  0.0745,
    0.0961,  0.0553,  0.0487, -0.3991,  0.0980,  0.1010,
    0.0895,  0.0482, -0.3750,  0.1033,  0.0644,  0.0696,
    0.0982,  0.0708, -0.3932,  0.0675,  0.0966,  0.0600,
    0.0470, -0.3696,  0.0767,  0.0857,  0.0666,  0.0937,
  };
  std::array<float, N * N> expected_trans_grad = {
    0.1784,  0.1519,  0.1559,  0.1754,  0.1567,  0.1411,
    0.1640,  0.1416, -0.7458,  0.1671,  0.1503, -0.3198,
    0.1551,  0.1414, -0.3100, -0.3042,  0.1438,  0.1217,
    0.1652, -0.2991,  0.1363, -0.7343, -0.2939,  0.1284,
    0.1729,  0.1485,  0.1542,  0.1638, -0.2928,  0.1359,
    0.1628, -0.3105,  0.1336, -0.3006,  0.1438,  0.1213,
  };
  // clang-format on

  auto loss = linseg({input_af, target_af}).front();
  std::vector<float> loss_host(B);
  loss.host(loss_host.data());
  for (int i = 0; i < B; i++) {
    ASSERT_NEAR(loss_host[i], expected_loss[i], 1e-3);
  }

  loss.backward();
  auto input_grad = input_af.grad().array();
  checkZero(input_grad - af::array(N, T, B, expected_input_grad.data()), 1e-4);
  auto trans_grad = linseg.param(0).grad().array();
  checkZero(trans_grad - af::array(N, N, expected_trans_grad.data()), 1e-4);
}

TEST(CriterionTest, AsgSerialization) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = "/tmp/" + userstr + "_test.mdl";
  int N = 500;

  auto asg = std::make_shared<AutoSegmentationCriterion>(N);
  fl::save(path, asg);

  std::shared_ptr<AutoSegmentationCriterion> asg2;
  fl::load(path, asg2);

  checkZero((asg->param(0) - asg2->param(0)).array(), 1e-4);

  auto input = af::randu(N, 200, 2);
  auto target = af::clamp(af::randu(100, 2, af::dtype::s32), 0, N - 1);
  checkZero((asg->param(0) - asg2->param(0)).array(), 1e-4);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

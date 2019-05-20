/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Binary for comparing output of ASG implementations. Use the `generate`
 * command to create random inputs, `baseline` to run the criterion and save a
 * baseline, and `compare` to re-run the criterion and compare against baseline.
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>

#include <flashlight/flashlight.h>

#include "criterion/AutoSegmentationCriterion.h"

using namespace w2l;

namespace {

constexpr int B = 100;
constexpr int N = 30;
constexpr int T = 1500;
// intentionally set L > T to check that ASG truncates long targets
constexpr int L = 2000;

std::random_device rd;

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " [COMMAND] [FILES]...\n";
  std::cerr << "  generate input_file\n";
  std::cerr << "  baseline input_file baseline_file\n";
  std::cerr << "  compare  input_file baseline_file" << std::endl;
  std::exit(1);
}

struct CriterionOutput {
  af::array loss;
  af::array inputGrad;
  af::array transGrad;
};

CriterionOutput
run(const af::array& input, const af::array& target, const af::array& trans) {
  fl::Variable inputVar(input, true);
  fl::Variable targetVar(target, false);
  fl::Variable transVar(trans, true);

  AutoSegmentationCriterion crit(N, CriterionScaleMode::TARGET_SZ_SQRT);
  crit.setParams(transVar, 0);
  auto loss = crit.forward({inputVar, targetVar}).front();
  loss.backward();

  CriterionOutput result;
  result.loss = loss.array();
  result.inputGrad = inputVar.grad().array();
  result.transGrad = transVar.grad().array();
  return result;
}

// Discrepancy value of 1 corresponds to `allclose(a, b, rtol=1e-5, atol=1e-7)`
// just barely returning true.
double discrepancy(const af::array& a, const af::array& b) {
  const auto& ad = a.as(f64);
  const auto& bd = b.as(f64);
  return af::max<double>(af::abs(ad - bd) / (1e-7 + 1e-5 * af::abs(bd)));
}

void printDiscrepancies(
    const std::string& prefix,
    const af::array& compare,
    const af::array& baseline) {
  std::cerr << prefix << "discrepancy=" << std::setprecision(17)
            << discrepancy(compare, baseline);
  // Since discrepancy() uses af::max<T> which passes over NaNs,
  // we check for NaN discrepancies manually.
  auto compareNaN = af::isNaN(compare);
  auto baselineNaN = af::isNaN(baseline);
  if (af::anyTrue<bool>(compareNaN && !baselineNaN)) {
    std::cerr << " (warning: compare has NaNs where baseline does not)";
  } else if (af::anyTrue<bool>(compareNaN && baselineNaN)) {
    std::cerr << " (warning: both baseline and compare have NaNs)";
  } else if (af::anyTrue<bool>(baselineNaN)) {
    std::cerr << " (warning: baseline has NaNs where compare does not)";
  }
  std::cerr << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0]);
  }

  std::string command = argv[1];

  if (command == "generate") {
    if (argc != 3) {
      usage(argv[0]);
    }

    std::seed_seq seeds({rd(), rd(), rd(), rd()});
    std::mt19937 rng(seeds);

    // generate random target sizes
    std::vector<int> targetSize(B);
    for (int b = 0; b < B; ++b) {
      // ensure we have a sample with targetSize=1 and targetSize=L
      targetSize[b] = (b == B - 1) ? L : (b == B - 2) ? 1 : (1 + rng() % L);
    }
    std::shuffle(targetSize.begin(), targetSize.end(), rng);

    // generate random targets with the above sizes
    std::vector<int> targetHost(B * L);
    for (int b = 0; b < B; ++b) {
      auto* targetCur = &targetHost[b * L];
      for (int i = 0; i < targetSize[b]; ++i) {
        targetCur[i] = rng() % N;
      }
      for (int i = targetSize[b]; i < L; ++i) {
        targetCur[i] = -1;
      }
    }

    uint64_t afSeed = rng();
    afSeed <<= 32;
    afSeed ^= rng();
    af::setSeed(afSeed);

    auto input = af::randn(N, T, B);
    auto target = af::array(L, B, targetHost.data());
    auto trans = af::randn(N, N);
    fl::save(argv[2], input, target, trans);
    std::cerr << "input generated" << std::endl;
  } else if (command == "baseline") {
    if (argc != 4) {
      usage(argv[0]);
    }

    af::array input, target, trans;
    fl::load(argv[2], input, target, trans);
    auto out = run(input, target, trans);
    fl::save(argv[3], out.loss, out.inputGrad, out.transGrad);
    std::cerr << "baseline saved" << std::endl;
  } else if (command == "compare") {
    if (argc != 4) {
      usage(argv[0]);
    }

    af::array input, target, trans;
    fl::load(argv[2], input, target, trans);
    CriterionOutput out0;
    fl::load(argv[3], out0.loss, out0.inputGrad, out0.transGrad);
    auto out = run(input, target, trans);
    std::cerr << "Computing discrepancies vs. 1e-5 rel + 1e-7 abs tolerance\n";
    printDiscrepancies("loss: ", out.loss, out0.loss);
    printDiscrepancies("inputGrad: ", out.inputGrad, out0.inputGrad);
    printDiscrepancies("transGrad: ", out.transGrad, out0.transGrad);
  } else {
    usage(argv[0]);
  }
}

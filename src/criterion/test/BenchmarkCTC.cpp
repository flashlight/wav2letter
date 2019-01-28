/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/flashlight.h>

#include <iomanip>
#include <iostream>

#include <arrayfire.h>
#include <array>

#include "criterion/criterion.h"

using namespace fl;
using namespace w2l;

int main() {
  af::info();
  af::setDevice(1);
  auto ctc = ConnectionistTemporalClassificationCriterion();

  int N = 30, T = 487, L = 34, B = 10;

  auto input = Variable(af::log(af::randu(N, T, B)), true);

  auto t =
      af::abs(af::randu(L, B, af::dtype::s32)).as(af::dtype::s32) % (N - 2);

  for (int i = 0; i < B; ++i) {
    int r = rand() % (L / 2);
    t(af::seq(L / 2 + r, af::end), i) = -1;
  }

  Variable target(t, false);
  int ntimes = 50;
  Variable b = ctc.forward({input, target}).front();
  Variable gradoutput = Variable(af::randu(b.dims()) * 2 - 2, false);
  for (int i = 0; i < 5; ++i) {
    b = ctc.forward({input, target}).front();
    b.backward();
  }
  af::sync();
  auto s = af::timer::start();
  for (int i = 0; i < ntimes; ++i) {
    b = ctc.forward({input, target}).front();
    b.backward(gradoutput);
  }
  af::sync();
  auto e = af::timer::stop(s);
  std::cout << "Total time (fwd+bwd pass) " << std::setprecision(5)
            << e * 1000.0 / ntimes << " msec" << std::endl;
  return 0;
}

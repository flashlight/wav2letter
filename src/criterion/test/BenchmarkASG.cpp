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
  af::setDevice(1);
  int N = 30, T = 487, L = 34, B = 20;

  auto asg = AutoSegmentationCriterion(N);

  auto input = Variable(af::randu(N, T, B) * 2 - 1, true);

  auto target = Variable(
      af::abs(af::randu(L, B, af::dtype::s32)).as(af::dtype::s32) % (N - 1),
      false);

  int ntimes = 50;
  Variable b = asg.forward({input, target}).front();
  Variable gradoutput = Variable(af::randu(b.dims()) * 2 - 2, false);
  for (int i = 0; i < 5; ++i) {
    b = asg.forward({input, target}).front();
    b.backward();
  }
  af::sync();
  auto s = af::timer::start();
  for (int i = 0; i < ntimes; ++i) {
    b = asg.forward({input, target}).front();
    b.backward(gradoutput);
  }
  af::sync();
  auto e = af::timer::stop(s);
  std::cout << "Total time (fwd+bwd pass) " << std::setprecision(5)
            << e * 1000.0 / ntimes << " msec" << std::endl;
  return 0;
}

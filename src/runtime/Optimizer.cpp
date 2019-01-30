/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "runtime/Optimizer.h"

#include <glog/logging.h>

namespace w2l {

std::shared_ptr<fl::FirstOrderOptimizer> initOptimizer(
    const std::shared_ptr<fl::Module>& net,
    const std::string& optimizer,
    double lr,
    double momentum,
    double weightdecay) {
  std::shared_ptr<fl::FirstOrderOptimizer> opt;
  if (optimizer == kSGDoptimizer) {
    opt = std::make_shared<fl::SGDOptimizer>(
        net->params(), lr, momentum, weightdecay);
  } else if (optimizer == kAdamOptimizer) {
    opt = std::make_shared<fl::AdamOptimizer>(
        net->params(),
        lr,
        FLAGS_adambeta1,
        FLAGS_adambeta2,
        FLAGS_optimepsilon,
        weightdecay);
  } else if (optimizer == kRMSPropOptimizer) {
    opt = std::make_shared<fl::RMSPropOptimizer>(
        net->params(), lr, FLAGS_optimrho, FLAGS_optimepsilon, weightdecay);
  } else {
    LOG(FATAL) << "Optimizer option " << optimizer << " not implemented";
  }

  return opt;
}
} // namespace w2l

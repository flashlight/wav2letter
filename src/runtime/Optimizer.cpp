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
    const std::vector<std::shared_ptr<fl::Module>>& nets,
    const std::string& optimizer,
    double lr,
    double momentum,
    double weightdecay) {
  if (nets.size() == 0) {
    throw std::invalid_argument("no network for initializing the optimizer");
  }

  std::vector<fl::Variable> params;
  for (const auto& n : nets) {
    auto p = n->params();
    params.insert(params.end(), p.begin(), p.end());
  }

  std::shared_ptr<fl::FirstOrderOptimizer> opt;
  if (optimizer == kSGDOptimizer) {
    opt = std::make_shared<fl::SGDOptimizer>(params, lr, momentum, weightdecay);
  } else if (optimizer == kAdamOptimizer) {
    opt = std::make_shared<fl::AdamOptimizer>(
        params,
        lr,
        FLAGS_adambeta1,
        FLAGS_adambeta2,
        FLAGS_optimepsilon,
        weightdecay);
  } else if (optimizer == kRMSPropOptimizer) {
    opt = std::make_shared<fl::RMSPropOptimizer>(
        params, lr, FLAGS_optimrho, FLAGS_optimepsilon, weightdecay);
  } else if (optimizer == kAdadeltaOptimizer) {
    opt = std::make_shared<fl::AdadeltaOptimizer>(
        params, lr, FLAGS_optimrho, FLAGS_optimepsilon, weightdecay);
  } else if (optimizer == kAdagradOptimizer) {
    opt =
        std::make_shared<fl::AdagradOptimizer>(params, lr, FLAGS_optimepsilon);
  } else if (optimizer == kAMSgradOptimizer) {
    opt = std::make_shared<fl::AMSgradOptimizer>(
        params,
        lr,
        FLAGS_adambeta1,
        FLAGS_adambeta2,
        FLAGS_optimepsilon,
        weightdecay);

  } else if (optimizer == kNovogradOptimizer) {
    opt = std::make_shared<fl::NovogradOptimizer>(
        params,
        lr,
        FLAGS_adambeta1,
        FLAGS_adambeta2,
        FLAGS_optimepsilon,
        weightdecay);
  } else {
    LOG(FATAL) << "Optimizer option " << optimizer << " not implemented";
  }

  return opt;
}

} // namespace w2l

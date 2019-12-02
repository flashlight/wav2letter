/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/decoder/Utils.h"

namespace w2l {

bool isValidCandidate(
    double& bestScore,
    const double score,
    const double beamThreshold) {
  if (score >= bestScore) {
    bestScore = score;
  }

  return score >= bestScore - beamThreshold;
}

} // namespace w2l

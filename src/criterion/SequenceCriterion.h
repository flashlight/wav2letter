/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace fl {

class SequenceCriterion : public Loss {
 public:
  virtual af::array viterbiPath(const af::array& input) = 0;

 private:
  FL_SAVE_LOAD_WITH_BASE(Loss)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::SequenceCriterion)

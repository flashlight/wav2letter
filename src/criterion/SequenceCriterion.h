/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

class SequenceCriterion : public fl::Container {
 public:
  virtual af::array viterbiPath(const af::array& input) = 0;

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::SequenceCriterion)

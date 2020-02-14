/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>

#include <flashlight/flashlight.h>

namespace w2l {

class SequenceCriterion : public fl::Container {
 public:
  // Find the most likely path through input using viterbi algorithm
  // https://en.wikipedia.org/wiki/Viterbi_algorithm
  virtual af::array viterbiPath(const af::array& input) = 0;

  // Finds the most likely path using viterbi algorithm that is constrained to
  // go through target
  virtual af::array viterbiPath(
      const af::array& input,
      const af::array& target) {
    throw std::runtime_error("Not implmented");
    return af::array();
  }

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::SequenceCriterion)

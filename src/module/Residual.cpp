/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <stdexcept>

#include "module/Residual.h"

using namespace fl;

namespace w2l {

Residual::Residual(int num_layers)
    : shortcut_(num_layers + 1), reverseShortcut_(num_layers) {}

void Residual::addShortcut(int startId, int endId) {
  if (startId < 0 || startId >= shortcut_.size() || endId <= 0 ||
      endId > shortcut_.size() + 1 || endId - startId <= 1) {
    throw std::invalid_argument("invalid skip connection");
  }

  shortcut_[endId - 1].insert(startId);
  reverseShortcut_[startId].insert(endId - 1);
}

std::vector<Variable> Residual::forward(const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("Residual module expects only one input");
  }
  return {forward(inputs[0])};
}

Variable Residual::forward(const Variable& input) {
  Variable output = input;
  std::vector<Variable> outputs(shortcut_.size(), Variable());
  outputs[0] = input;

  for (int idx = 0; idx < modules_.size(); ++idx) {
    for (auto id : shortcut_[idx]) {
      output = output + outputs[id];
    }

    output = modules_[idx]->forward({output}).front();
    outputs[idx + 1] = output;
  }

  for (auto id : shortcut_.back()) {
    output = output + outputs[id];
  }

  return output;
}

std::string Residual::prettyString() const {
  std::ostringstream ss;
  ss << "Residual Block";
  for (int idx = 0; idx <= modules_.size(); ++idx) {
    ss << "\n\t" << idx << ": ";
    if (idx == 0) {
      ss << "Input";
    } else {
      ss << modules_[idx - 1]->prettyString();
    }

    if (idx < reverseShortcut_.size() && reverseShortcut_[idx].size() > 0) {
      ss << ", skip connection to ";
      for (auto i : reverseShortcut_[idx]) {
        if (i < reverseShortcut_.size()) {
          ss << "layer" << i + 1;
        } else {
          ss << "output";
        }
        ss << " ";
      }
    }
  }

  return ss.str();
}

} // namespace w2l

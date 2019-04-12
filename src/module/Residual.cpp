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

void Residual::addScale(int beforeLayer, float scale) {
  int nLayers = modules_.size() - projectionsIndices_.size();
  if (beforeLayer < 1 || beforeLayer > nLayers + 1) {
    throw std::invalid_argument(
        "Residual: invalid layer index " + std::to_string(beforeLayer) +
        " before which apply the scaling");
  }
  if (scales_.find(beforeLayer - 1) != scales_.end()) {
    throw std::invalid_argument(
        "Residual: scaling before layer " + std::to_string(beforeLayer) +
        " was already added; adding only once is allowed");
  }
  scales_[beforeLayer - 1] = scale;
}

void Residual::checkShortcut(int fromLayer, int toLayer) {
  int nLayers = modules_.size() - projectionsIndices_.size();

  if (fromLayer < 0 || fromLayer >= nLayers || toLayer <= 0 ||
      toLayer > nLayers + 2 || toLayer - fromLayer <= 1) {
    throw std::invalid_argument(
        "Residual: invalid skip connection; check fromLayer=" +
        std::to_string(fromLayer) + " and toLayer=" + std::to_string(toLayer) +
        " parameters. They are out of range of added layers");
  }
  if (shortcut_.find(toLayer - 1) != shortcut_.end() &&
      shortcut_[toLayer - 1].find(fromLayer) != shortcut_[toLayer - 1].end()) {
    throw std::invalid_argument(
        "Residual: skip connection for fromLayer " + std::to_string(fromLayer) +
        " to toLayer " + std::to_string(toLayer) + " is already added");
  }
}

void Residual::processShortcut(
    int fromLayer,
    int toLayer,
    int projectionIndex) {
  shortcut_[toLayer - 1].insert({fromLayer, projectionIndex});
}

void Residual::addShortcut(int fromLayer, int toLayer) {
  // fromLayer: 0, .., nLayers_ - 1; toLayer: 1, 2, .., nLayers_ + 1
  // toLayer - fromLayer > 1 (avoid adding skip connection
  // from layer K to layer  K+1)
  checkShortcut(fromLayer, toLayer);
  processShortcut(fromLayer, toLayer, -1);
}

Variable Residual::applyScale(const Variable& input, const int layerIndex) {
  float scale =
      scales_.find(layerIndex) != scales_.end() ? scales_[layerIndex] : 1.;
  return input * scale;
}

std::vector<Variable> Residual::forward(const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("Residual module expects only one input");
  }
  return {forward(inputs[0])};
}

Variable Residual::forward(const Variable& input) {
  Variable output = input;
  int nLayers = modules_.size() - projectionsIndices_.size();
  std::vector<Variable> outputs(nLayers + 1, Variable());
  outputs[0] = input;

  int moduleIndex = 0, layerIndex = 0;

  while (layerIndex < nLayers) {
    while (projectionsIndices_.find(moduleIndex) != projectionsIndices_.end()) {
      moduleIndex++;
    }
    if (shortcut_.find(layerIndex) != shortcut_.end()) {
      for (const auto& shortcut : shortcut_[layerIndex]) {
        Variable connectionOut = outputs[shortcut.first];
        if (shortcut.second != -1) {
          connectionOut = modules_[shortcut.second]
                              ->forward({outputs[shortcut.first]})
                              .front();
        }
        output = output + connectionOut;
      }
    }
    output = modules_[moduleIndex]
                 ->forward({applyScale(output, layerIndex)})
                 .front();
    outputs[layerIndex + 1] = output;
    layerIndex++;
    moduleIndex++;
  }
  if (shortcut_.find(nLayers) != shortcut_.end()) {
    for (const auto& shortcut : shortcut_[nLayers]) {
      Variable connectionOut = outputs[shortcut.first];
      if (shortcut.second != -1) {
        connectionOut = modules_[shortcut.second]
                            ->forward({outputs[shortcut.first]})
                            .front();
      }
      output = output + connectionOut;
    }
  }
  return applyScale(output, nLayers);
}

std::string Residual::prettyString() const {
  std::ostringstream ss;
  // prepare inverted residual skip connection
  std::unordered_map<int, std::unordered_map<int, int>>
      reverseShortcut; // start -> end
  for (const auto& shortcut : shortcut_) {
    for (const auto& value : shortcut.second) {
      reverseShortcut[value.first].insert({shortcut.first, value.second});
    }
  }

  int nLayers = modules_.size() - projectionsIndices_.size();
  int moduleIndex = -1, layerIndex = 0;
  std::unordered_map<int, float>::const_iterator scaleIt;

  while (layerIndex <= nLayers) {
    ss << "\n\tRes(" << layerIndex << "): ";
    if (layerIndex == 0) {
      ss << "Input";
    } else {
      while (projectionsIndices_.find(moduleIndex) !=
             projectionsIndices_.end()) {
        moduleIndex++;
      }
      ss << modules_[moduleIndex]->prettyString();
    }

    scaleIt = scales_.find(layerIndex);
    if (scaleIt != scales_.end()) {
      ss << " with scale (before layer is applied) " << scaleIt->second << ";";
    }

    if (reverseShortcut.find(layerIndex) != reverseShortcut.end() &&
        reverseShortcut[layerIndex].size() > 0) {
      ss << "; skip connection to ";
      for (auto shortcut : reverseShortcut[layerIndex]) {
        if (shortcut.first < nLayers) {
          ss << "layer Res(" << shortcut.first + 1 << ")";
        } else {
          ss << "output";
        }
        if (shortcut.second != -1) {
          ss << " with transformation: "
             << modules_[shortcut.second]->prettyString() << ";";
        }
        ss << " ";
      }
    }
    layerIndex++;
    moduleIndex++;
  }
  ss << "\n\tRes(" << nLayers + 1 << "): Output;";
  scaleIt = scales_.find(nLayers + 1);
  if (scaleIt != scales_.end()) {
    ss << " with scale (before layer is applied) " << scaleIt->second << ";";
  }

  return ss.str();
}

} // namespace w2l

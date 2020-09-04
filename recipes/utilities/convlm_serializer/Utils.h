/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <flashlight/flashlight.h>
#include <string>
#include <vector>

struct ConvLMParamState {
  const std::string moduleName;
  const std::string layerName;
  const std::string paramName;
  af::array weights;
};

std::vector<ConvLMParamState> loadModelStates(const std::string& weightFile);

void loadLayer(
    std::vector<ConvLMParamState>& states,
    std::vector<int>& layerIndices,
    std::shared_ptr<fl::Module> mainModule,
    std::shared_ptr<fl::Module> layer,
    std::string layerName,
    int paramIdx);

void loadConvLM(
    std::shared_ptr<fl::Module>& network,
    std::shared_ptr<fl::BinaryModule>& criterion,
    const std::string& arcFile,
    const std::string& weightFile,
    int outputTokensDim,
    const std::vector<int>& adaptiveTail = std::vector<int>(),
    int inputSizeAdaptiveSoftmax = 0);

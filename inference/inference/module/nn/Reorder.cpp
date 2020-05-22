/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Reorder.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

Reorder::Reorder() {
}

std::shared_ptr<ModuleProcessingState> Reorder::start(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  assert(input->buffers().size() == 1);
  return input;
}

std::shared_ptr<ModuleProcessingState> Reorder::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  assert(input->buffers().size() == 1);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  if (inputBuf->size<float>()==0)
    return input;

  // for each timestep (dim 0) transpose 2d matrix
  dim_t batches = inputBuf->dim[0];
  dim_t rows = inputBuf->dim[1];
  dim_t cols = inputBuf->dim[2];
  std::vector<float> in(inputBuf->data<float>(), inputBuf->data<float>() + inputBuf->size<float>());
  float* out = inputBuf->data<float>();
  for (size_t b = 0; b < batches; ++b) {
    int64_t start = b * rows * cols;
    for (size_t c = 0; c < cols; ++c) {
      for (size_t r = 0; r < rows; ++r) {
        out[start + c * rows + r] = in[start + r * cols + c];
      }
    }
  }
  inputBuf->dim = dim4(inputBuf->dim[0], inputBuf->dim[2], inputBuf->dim[1]);

  return input;
}

std::string Reorder::debugString() const {
  std::stringstream ss;
  ss << "Reorder:{}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l

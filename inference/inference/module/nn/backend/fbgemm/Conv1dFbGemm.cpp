/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/backend/fbgemm/Conv1dFbGemm.h"

#include <sstream>
#include <stdexcept>

#include "inference/common/IOBuffer.h"

namespace w2l {
namespace streaming {

Conv1dFbGemm::Conv1dFbGemm(
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    int rightPadding,
    int leftPadding,
    int groups,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias)
    : Conv1d(
          inChannels,
          outChannels,
          kernelSize,
          stride,
          rightPadding,
          leftPadding,
          groups),
      bias_(bias) {
  if (!weights || !bias || weights->type_ != DataType::FLOAT ||
      bias->type_ != DataType::FLOAT) {
    std::stringstream ss;
    ss << "Invalid argument at"
       << " Conv1dFbGemm::Conv1dFbGemm(groups=" << groups
       << "inChannels=" << inChannels << " outChannels=" << outChannels
       << " kernelSize=" << kernelSize << " stride=" << stride
       << " rightPadding=" << rightPadding << " rightPadding=" << rightPadding
       << " weights=" << (weights ? weights->debugString() : "nullptr")
       << " bias=" << (bias ? bias->debugString() : "nullptr") << ")";
    throw std::invalid_argument(ss.str());
  }
  init(weights);
}

// Used for serialization loading only. Initialize using temporary valid bogus
// values.
Conv1dFbGemm::Conv1dFbGemm() : Conv1d(1, 1, 1, 1, 1, 1, 1) {}

void Conv1dFbGemm::init(std::shared_ptr<ModuleParameter> weights) {
  constexpr float alpha = 1.0;
  packedWeights_ = std::make_shared<fbgemm::PackedGemmMatrixFP16>(
      fbgemm::matrix_op_t::Transpose,
      (inChannels_ / groups_) * kernelSize_, // k
      (outChannels_ / groups_), // n
      alpha,
      weights->buffer_.data<float>());
}

std::string Conv1dFbGemm::debugString() const {
  std::stringstream ss;
  ss << "Conv1dFbGemm:{base=" << Conv1d::debugString() << " packedWeights_="
     << (packedWeights_ ? w2l::streaming::debugString(*packedWeights_)
                        : "nullptr")
     << "} bias_=" << (bias_ ? bias_->debugString() : "nullptr") << "}";
  return ss.str();
}

std::shared_ptr<ModuleProcessingState> Conv1dFbGemm::start(
    std::shared_ptr<ModuleProcessingState> input) {
  if (leftPadding_ > 0) {
    assert(input);
    assert(!input->buffers().empty());
    std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
    assert(inputBuf);

    IOBuffer tempBuf = *inputBuf;
    inputBuf->clear();
    inputBuf->writeZero<float>(leftPadding_ * inChannels_);
    inputBuf->write<float>(tempBuf.data<float>(), tempBuf.size<float>());
  }
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> Conv1dFbGemm::finish(
    std::shared_ptr<ModuleProcessingState> input) {
  if (rightPadding_ > 0) {
    assert(input);
    assert(!input->buffers().empty());
    std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
    assert(inputBuf);
    inputBuf->writeZero<float>(rightPadding_ * inChannels_);
  }
  return run(input);
}

namespace {
void unfoldDepthwise(
    float* dst,
    const float* src,
    const int inChannels,
    const int kernelSize,
    const int stride,
    const int outDim,
    const int depth) {
  for (int t = 0; t < outDim; ++t) {
    for (int d = 0; d < depth; ++d) {
      for (int ts = 0; ts < kernelSize; ++ts) {
        const float* ptr =
            src + (ts + t * stride) * depth * inChannels + d * inChannels;
        std::copy(ptr, ptr + inChannels, dst);
        dst += inChannels;
      }
    }
  }
}
} // namespace

std::shared_ptr<ModuleProcessingState> Conv1dFbGemm::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  assert(!input->buffers().empty());
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  assert(!output->buffers().empty());

  const int nInFrames = inputBuf->size<float>() / inChannels_;
  if (nInFrames < kernelSize_) {
    return output;
  }

  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  int nOutFrames = (nInFrames - kernelSize_) / stride_ + 1;
  int outSize = nOutFrames * outChannels_;
  int consumedSize = nOutFrames * stride_ * inChannels_;

  outputBuf->ensure<float>(outSize);
  auto* outPtr = outputBuf->tail<float>();
  for (int i = 0; i < nOutFrames * groups_; ++i) {
    std::copy_n(
        bias_->buffer_.data<float>(),
        outChannels_ / groups_,
        outPtr + i * (outChannels_ / groups_));
  }

  if (!memoryManager_) {
    throw std::invalid_argument("null memoryManager_ at Conv1dFbGemm::run()");
  }
  auto workspace = memoryManager_->makeShared<float>(
      (kernelSize_ * inChannels_ * outChannels_ * nOutFrames) / groups_);
  assert(workspace);

  unfoldDepthwise(
      workspace.get() /* dst */,
      inputBuf->data<float>() /* src */,
      inChannels_ / groups_,
      kernelSize_,
      stride_,
      nOutFrames,
      groups_);

  constexpr float beta = 1.0;
  cblas_gemm_compute(
      fbgemm::matrix_op_t::NoTranspose,
      nOutFrames * groups_,
      workspace.get(),
      *packedWeights_,
      beta,
      outPtr);

  outputBuf->move<float>(outSize);
  inputBuf->consume<float>(consumedSize);
  return output;
}

std::shared_ptr<Conv1d> createConv1d(
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    const std::pair<int, int> padding,
    int groups,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias) {
  return std::make_shared<Conv1dFbGemm>(
      inChannels,
      outChannels,
      kernelSize,
      stride,
      padding.second,
      padding.first,
      groups,
      weights,
      bias);
}

} // namespace streaming
} // namespace w2l

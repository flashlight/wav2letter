/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Container.h"
#include "flashlight/fl/nn/modules/LayerNorm.h"
#include "flashlight/fl/nn/modules/Linear.h"
#include "flashlight/fl/nn/modules/Module.h"

using namespace fl;

namespace w2l {
namespace cpc {

/**
 * A module which implements a Transformer.
 *
 * For details, see [Vaswani et al
 * (2017)](https://arxiv.org/abs/1706.03762).
 *
 * This module also supports layer drop regularization, as introduced in
 * [Fan et al (2019)](https://arxiv.org/abs/1909.11556).
 *
 * Forward takes {previous step[optionally], input, padMask}
 * previous step is used in for the decoder phase, previous output with size
 * CxT'xBx1 Input dimension at forward is assumed to be CxTxBx1, where C is the
 * number of features, T the sequence length and B the batch size.
 * padMask is with T''xB sizes (T'' will be af::resize to the input size)
 * padMask should be empty if "previous step" is provided (in the decoder phase)
 * padMask is expected to have "1" on the normal positions and "0" on the padded
 * positions
 *
 * @param modelDim input embedding dimension
 * @param headDim dimension of each head
 * @param mlpDim dimension of the feed-forward layers
 * @param nHeads number of heads
 * @param bptt size for learnt relative positional embedding matrix (2 * bptt -
 * 1) * nHeads
 * @param pDropout dropout probability
 * @param pLayerdrop layer dropout probability
 * @param useMask mask or not future in the computations
 * if true then don't use future (for example for autoregressive language models
 * or for decoder part in the encoder-decoder transformer models)
 * @param preLN apply layer normalization before or after residual connection
 */
class TransformerCPC : public Container {
 public:
  TransformerCPC(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t bptt,
      float pDropout,
      float pLayerdrop,
      bool useMask = false,
      bool preLN = false,
      double layerNormEps = 1e-5);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;
  std::string prettyString() const override;

 private:
  int32_t nHeads_;
  int32_t bptt_;
  double pDropout_;
  double pLayerdrop_;
  bool useMask_;
  bool preLN_;
  double layerNormEps_;
  std::shared_ptr<Linear> w1_, w2_, wq_, wk_, wv_, wf_;
  std::shared_ptr<LayerNorm> norm1_, norm2_;

  Variable mlp(const Variable& input);
  Variable getMask(int32_t n, bool cache = false);
  Variable selfAttention(const std::vector<Variable>& input);

  FL_SAVE_LOAD_WITH_BASE(
      Container,
      w1_,
      w2_,
      wq_,
      wk_,
      wv_,
      wf_,
      norm1_,
      norm2_,
      nHeads_,
      pDropout_,
      pLayerdrop_,
      bptt_,
      useMask_,
      preLN_)

  TransformerCPC();
};

} // namespace cpc
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::cpc::TransformerCPC);

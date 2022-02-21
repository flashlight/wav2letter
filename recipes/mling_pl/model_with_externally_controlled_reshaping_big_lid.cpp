/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Adapted from Tatiana's ctc_letters_st3_ls100h_slimIPL_dp03_dyndp architecture
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/nn/modules/modules.h"

namespace slimIPL {
class myModel : public fl::Container {
 public:
  myModel(int64_t nFeature, int64_t nLabel) {
    convFrontend_->add(
        std::make_shared<fl::View>(af::dim4(-1, 1, nFeature, 0)));
    // Time x 1 x nFeature x Batch
    std::vector<int> lnDims = {0, 1, 2};
    convFrontend_->add(std::make_shared<fl::LayerNorm>(lnDims));
    convFrontend_->add(
        // std::make_shared<fl::Conv2D>(nFeature, 1536, 7, 1, 3, 1, -1, 0, 1,
        // 1));
        std::make_shared<fl::Conv2D>(nFeature, 3072, 7, 1, 3, 1, -1, 0, 1, 1));
    convFrontend_->add(std::make_shared<fl::GatedLinearUnit>(2));
    convFrontend_->add(std::make_shared<fl::Dropout>(0.3));
    convFrontend_->add(std::make_shared<fl::Reorder>(2, 0, 3, 1));
    // nFeature x Time x Batch x 1
    add(convFrontend_);
    for (int trIdx = 0; trIdx < 36; trIdx++) {
      auto layer = std::make_shared<fl::Transformer>(
          // 768, 192, 3072, 4, 920, 0.3, 0.3, false, false);
          1536,
          384,
          6144,
          4,
          920,
          0.3,
          0.3,
          false,
          false);
      transformers_.push_back(layer);
      add(layer);
    }
    // linear_ = std::make_shared<fl::Linear>(768, nLabel);
    linear_ = std::make_shared<fl::Linear>(1536, nLabel);
    add(linear_);

    int nLanguages = 60;
    // LID_head_ = std::make_shared<fl::Linear>(768, nLanguages);
    LID_head_ = std::make_shared<fl::Linear>(1536, nLanguages);
    add(LID_head_);
  }

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override {
    auto out = input[0];
    auto xSizes = input[1].array();
    float reshaping_factor = 1;
    if (input.size() > 2) {
      reshaping_factor = af::sum<float>(input[2].array());
    }
    float dp = -1;
    if (input.size() > 3) {
      dp = af::sum<float>(input[3].array());
    }
    // expected input dims T x C x 1 x B
    out = convFrontend_->forward(out);
    ///////// reshape ////////
    int time_dim = 1, feat_dim = 0, other_dim = 3, batch_dim = 2;
    int old_B = out.dims(batch_dim);
    int old_T = out.dims(time_dim);
    int new_B = old_B;
    int new_T = old_T;
    int T_padded = old_T;
    if (reshaping_factor != 1) {
      new_T = ceil(reshaping_factor * old_T);
      new_T += old_B -
          (new_T % old_B); // add this chunk so that new_T is divisible by old_B
      new_B = ceil((float)(old_B * old_T) / (float)new_T);
      T_padded = (new_B * new_T) / old_B;
      std::vector<std::pair<int, int>> pad_amount;
      pad_amount.push_back(std::make_pair(0, 0));
      pad_amount.push_back(std::make_pair(0, T_padded - old_T));
      pad_amount.push_back(std::make_pair(0, 0));
      pad_amount.push_back(std::make_pair(0, 0));
      out = fl::padding(out, pad_amount, 0.0);
      out = fl::reorder(out, time_dim, batch_dim, feat_dim, other_dim);
      time_dim = 0, feat_dim = 2, other_dim = 3, batch_dim = 1;
      auto new_out_dims = out.dims();
      new_out_dims[time_dim] = new_T;
      new_out_dims[batch_dim] = new_B;
      out = fl::moddims(out, new_out_dims);
      out = fl::reorder(out, feat_dim, time_dim, batch_dim, other_dim);
      //   std::cout << "(reshaping)\n";
    } else {
      //   std::cout << "(not reshaping)\n";
    }
    // std::cout << "old_B: " << old_B << "\n";
    // std::cout << "old_T: " << old_T << "\n";
    // std::cout << "new_B: " << new_B << "\n";
    // std::cout << "new_T: " << new_T << "\n";
    // std::cout << "T_padded: " << T_padded << "\n";
    if (T_padded * old_B != new_T * new_B) {
      std::cout << "error, T_padded * old_B != new_T * new_B\n";
      exit(0);
    }
    //////////////////////////
    af::array inputNotPaddedSize(1, old_B, 1, 1);
    for (int bIdx = 0; bIdx < old_B; bIdx++) {
      inputNotPaddedSize(0, bIdx, 0, 0) = old_T;
    } // TODO: use actual xSizes here
    auto padMask = af::iota(af::dim4(T_padded, 1), af::dim4(1, old_B)) <
        af::tile(inputNotPaddedSize, T_padded, 1);
    padMask = af::moddims(padMask, af::dim4(new_T, new_B, 1, 1));
    for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
      // NOTE: not required for inference
      //   if (dp >= 0) {
      //     transformers_[trIdx]->setDropout(dp);
      //     transformers_[trIdx]->setLayerDropout(dp);
      //   }
      out = transformers_[trIdx]->forward({out, fl::noGrad(padMask)}).front();
    }
    ///////// reshape ////////
    if (reshaping_factor != 1) {
      time_dim = 1, feat_dim = 0, other_dim = 3, batch_dim = 2;
      out = fl::reorder(out, time_dim, batch_dim, feat_dim, other_dim);
      time_dim = 0, feat_dim = 2, other_dim = 3, batch_dim = 1;
      auto new_tr_out_dims = out.dims();
      new_tr_out_dims[time_dim] = T_padded;
      new_tr_out_dims[batch_dim] = old_B;
      out = fl::moddims(out, new_tr_out_dims);
      out = fl::reorder(out, feat_dim, time_dim, batch_dim, other_dim);
      out = out(af::span, af::seq(old_T), af::span, af::span);
    }
    //////////////////////////
    auto ctc_head_out = linear_->forward(out);
    auto LID_head_out = LID_head_->forward(out);
    LID_head_out = fl::mean(LID_head_out.as(f32), std::vector<int>{1}).as(f32);
    LID_head_out = fl::logSoftmax(LID_head_out, 0);
    return {
        ctc_head_out.as(input[0].type()),
        LID_head_out}; //.as(input[0].type())};
  }

  std::string prettyString() const override {
    std::ostringstream ss;
    ss << "Model myModel: ";
    ss << convFrontend_->prettyString() << "\n";
    ss << "(reshaping happens here)\n";
    for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
      ss << transformers_[trIdx]->prettyString() << "\n";
    }
    ss << "(inverse reshaping happens here)\n";
    ss << "CTC head: " << linear_->prettyString() << "\n";
    ss << "Language ID head: " << LID_head_->prettyString() << "\n";
    return ss.str();
  }

 private:
  myModel() = default;

  std::shared_ptr<fl::Sequential> convFrontend_{
      std::make_shared<fl::Sequential>()};
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;
  std::shared_ptr<fl::Linear> linear_;
  std::shared_ptr<fl::Linear> LID_head_;

  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      convFrontend_,
      transformers_,
      linear_,
      LID_head_)
};
} // namespace slimIPL

extern "C" fl::Module* createModule(int64_t nFeature, int64_t nLabel) {
  auto m = std::make_unique<slimIPL::myModel>(nFeature, nLabel);
  return m.release();
}

CEREAL_REGISTER_TYPE(slimIPL::myModel)

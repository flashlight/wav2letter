/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <queue>

#include "criterion/TransformerCriterion.h"

using namespace fl;

namespace w2l {

TransformerCriterion buildTransformerCriterion(
    int numClasses,
    int numLayers,
    float dropout,
    float layerdrop,
    int eosIdx) {
  std::shared_ptr<AttentionBase> attention;
  if (FLAGS_attention == w2l::kContentAttention) {
    attention = std::make_shared<ContentAttention>();
  } else if (FLAGS_attention == w2l::kKeyValueAttention) {
    attention = std::make_shared<ContentAttention>(true);
  } else if (FLAGS_attention == w2l::kNeuralContentAttention) {
    attention = std::make_shared<NeuralContentAttention>(FLAGS_encoderdim);
  } else if (FLAGS_attention == w2l::kSimpleLocationAttention) {
    attention = std::make_shared<SimpleLocationAttention>(FLAGS_attnconvkernel);
  } else if (FLAGS_attention == w2l::kLocationAttention) {
    attention = std::make_shared<LocationAttention>(
        FLAGS_encoderdim, FLAGS_attnconvkernel);
  } else if (FLAGS_attention == w2l::kNeuralLocationAttention) {
    attention = std::make_shared<NeuralLocationAttention>(
        FLAGS_encoderdim,
        FLAGS_attndim,
        FLAGS_attnconvchannel,
        FLAGS_attnconvkernel);
  } else {
    throw std::runtime_error("Unimplmented attention: " + FLAGS_attention);
  }

  std::shared_ptr<WindowBase> window;
  if (FLAGS_attnWindow == w2l::kNoWindow) {
    window = nullptr;
  } else if (FLAGS_attnWindow == w2l::kMedianWindow) {
    window = std::make_shared<MedianWindow>(
        FLAGS_leftWindowSize, FLAGS_rightWindowSize);
  } else if (FLAGS_attnWindow == w2l::kStepWindow) {
    window = std::make_shared<StepWindow>(
        FLAGS_minsil, FLAGS_maxsil, FLAGS_minrate, FLAGS_maxrate);
  } else if (FLAGS_attnWindow == w2l::kSoftWindow) {
    window = std::make_shared<SoftWindow>(
        FLAGS_softwstd, FLAGS_softwrate, FLAGS_softwoffset);
  } else if (FLAGS_attnWindow == w2l::kSoftPretrainWindow) {
    window = std::make_shared<SoftPretrainWindow>(FLAGS_softwstd);
  } else {
    throw std::runtime_error("Unimplmented window: " + FLAGS_attnWindow);
  }

  return TransformerCriterion(
      numClasses,
      FLAGS_encoderdim,
      eosIdx,
      FLAGS_maxdecoderoutputlen,
      numLayers,
      attention,
      window,
      FLAGS_trainWithWindow,
      FLAGS_labelsmooth,
      FLAGS_pctteacherforcing,
      dropout,
      layerdrop);
}

TransformerCriterion::TransformerCriterion(
    int nClass,
    int hiddenDim,
    int eos,
    int maxDecoderOutputLen,
    int nLayer,
    std::shared_ptr<AttentionBase> attention,
    std::shared_ptr<WindowBase> window,
    bool trainWithWindow,
    double labelSmooth,
    double pctTeacherForcing,
    double p_dropout,
    double p_layerdrop)
    : nClass_(nClass),
      eos_(eos),
      maxDecoderOutputLen_(maxDecoderOutputLen),
      nLayer_(nLayer),
      window_(window),
      trainWithWindow_(trainWithWindow),
      labelSmooth_(labelSmooth),
      pctTeacherForcing_(pctTeacherForcing) {
  add(std::make_shared<fl::Embedding>(hiddenDim, nClass));
  for (size_t i = 0; i < nLayer_; i++) {
    add(std::make_shared<Transformer>(
        hiddenDim,
        hiddenDim / 4,
        hiddenDim * 4,
        4,
        maxDecoderOutputLen,
        p_dropout,
        p_layerdrop,
        true));
  }
  add(std::make_shared<fl::Linear>(hiddenDim, nClass));
  add(attention);
  params_.push_back(fl::uniform(af::dim4{hiddenDim}, -1e-1, 1e-1));
}

std::vector<Variable> TransformerCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("Invalid inputs size");
  }
  const auto& input = inputs[0];
  const auto& target = inputs[1];

  Variable out, alpha;
  std::tie(out, alpha) = vectorizedDecoder(input, target);

  out = logSoftmax(out, 0);

  auto losses = moddims(
      sum(categoricalCrossEntropy(out, target, ReduceMode::NONE), {0}), -1);
  if (train_ && labelSmooth_ > 0) {
    size_t nClass = out.dims(0);
    auto smoothLoss = moddims(sum(out, {0, 1}), -1);
    losses = (1 - labelSmooth_) * losses - (labelSmooth_ / nClass) * smoothLoss;
  }

  return {losses, out};
}

// input : D x T x B
// target: U x B

std::pair<Variable, Variable> TransformerCriterion::vectorizedDecoder(
    const Variable& input,
    const Variable& target) {
  int U = target.dims(0);
  int B = target.dims(1);
  int T = input.isempty() ? 0 : input.dims(1);

  auto hy = tile(startEmbedding(), {1, 1, B});

  if (U > 1) {
    auto y = target(af::seq(0, U - 2), af::span);

    if (train_) {
      // TODO: other sampling strategies
      auto mask =
          Variable(af::randu(y.dims()) * 100 <= pctTeacherForcing_, false);
      auto samples =
          Variable((af::randu(y.dims()) * (nClass_ - 1)).as(s32), false);
      y = mask * y + (1 - mask) * samples;
    }

    auto yEmbed = embedding()->forward(y);
    hy = concatenate({hy, yEmbed}, 1);
  }

  Variable alpha, summaries;
  for (int i = 0; i < nLayer_; i++) {
    hy = layer(i)->forward(std::vector<Variable>({hy})).front();
  }

  if (!input.isempty()) {
    Variable windowWeight;
    if (window_ && (!train_ || trainWithWindow_)) {
      windowWeight = window_->computeWindowMask(U, T, B);
    }

    std::tie(alpha, summaries) =
        attention()->forward(hy, input, Variable(), windowWeight);

    hy = hy + summaries;
  }

  auto out = linearOut()->forward(hy);

  return std::make_pair(out, alpha);
}

af::array TransformerCriterion::viterbiPath(const af::array& input) {
  return viterbiPathBase(input, false).first;
}

std::pair<af::array, Variable> TransformerCriterion::viterbiPathBase(
    const af::array& input,
    bool /* TODO: saveAttn */) {
  bool wasTrain = train_;
  eval();
  std::vector<int> path;
  std::vector<Variable> alphaVec;
  Variable alpha;
  TS2SState state;
  Variable y, ox;
  af::array maxIdx, maxValues;
  int pred;

  for (int u = 0; u < maxDecoderOutputLen_; u++) {
    std::tie(ox, state) = decodeStep(Variable(input, false), y, state);
    max(maxValues, maxIdx, ox.array());
    maxIdx.host(&pred);
    // TODO: saveAttn

    if (pred == eos_) {
      break;
    }
    y = constant(pred, 1, s32, false);
    path.push_back(pred);
  }
  // TODO: saveAttn

  if (wasTrain) {
    train();
  }

  auto vPath = path.empty() ? af::array() : af::array(path.size(), path.data());
  return std::make_pair(vPath, alpha);
}

std::pair<Variable, TS2SState> TransformerCriterion::decodeStep(
    const Variable& xEncoded,
    const Variable& y,
    const TS2SState& inState) const {
  Variable hy;
  if (y.isempty()) {
    hy = tile(startEmbedding(), {1, 1, xEncoded.dims(2)});
  } else {
    hy = embedding()->forward(y);
  }

  // TODO: inputFeeding

  TS2SState outState;
  outState.step = inState.step + 1;
  for (int i = 0; i < nLayer_; i++) {
    if (inState.step == 0) {
      outState.hidden.push_back(hy);
      hy = layer(i)->forward(std::vector<Variable>({hy})).front();
    } else {
      auto tmp = std::vector<Variable>({inState.hidden[i], hy});
      outState.hidden.push_back(concatenate(tmp, 1));
      hy = layer(i)->forward(tmp).front();
    }
  }

  Variable windowWeight, alpha, summary;
  if (window_ && (!train_ || trainWithWindow_)) {
    windowWeight = window_->computeSingleStepWindow(
        Variable(), xEncoded.dims(1), xEncoded.dims(2), inState.step);
  }

  std::tie(alpha, summary) =
      attention()->forward(hy, xEncoded, Variable(), windowWeight);

  hy = hy + summary;

  auto out = linearOut()->forward(hy);
  return std::make_pair(out, outState);
}

std::pair<std::vector<std::vector<float>>, std::vector<TS2SStatePtr>>
TransformerCriterion::decodeBatchStep(
    const fl::Variable& xEncoded,
    std::vector<fl::Variable>& ys,
    const std::vector<TS2SState*>& inStates,
    const int /* attentionThreshold */,
    const float smoothingTemperature) const {
  int B = ys.size();

  for (int i = 0; i < B; i++) {
    if (ys[i].isempty()) {
      ys[i] = startEmbedding();
    } else {
      ys[i] = embedding()->forward(ys[i]);
    } // TODO: input feeding
    ys[i] = moddims(ys[i], {ys[i].dims(0), 1, -1});
  }
  Variable yBatched = concatenate(ys, 2); // D x 1 x B

  std::vector<TS2SStatePtr> outstates(B);
  for (int i = 0; i < B; i++) {
    outstates[i] = std::make_shared<TS2SState>();
    outstates[i]->step = inStates[i]->step + 1;
  }

  Variable outStateBatched;
  for (int i = 0; i < nLayer_; i++) {
    if (inStates[0]->step == 0) {
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(yBatched.slice(j));
      }
      yBatched = layer(i)->forward(std::vector<Variable>({yBatched})).front();
    } else {
      std::vector<Variable> statesVector(B);
      for (int j = 0; j < B; j++) {
        statesVector[j] = inStates[j]->hidden[i];
      }
      Variable inStateHiddenBatched = concatenate(statesVector, 2);
      auto tmp = std::vector<Variable>({inStateHiddenBatched, yBatched});
      auto tmp2 = concatenate(tmp, 1);
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(tmp2.slice(j));
      }
      yBatched = layer(i)->forward(tmp).front();
    }
  }

  Variable alpha, summary;
  yBatched = moddims(yBatched, {yBatched.dims(0), -1});
  std::tie(alpha, summary) =
      attention()->forward(yBatched, xEncoded, Variable(), Variable());
  alpha = reorder(alpha, 1, 0);
  yBatched = yBatched + summary;

  auto outBatched = linearOut()->forward(yBatched);
  outBatched = logSoftmax(outBatched / smoothingTemperature, 0);
  std::vector<std::vector<float>> out(B);
  for (int i = 0; i < B; i++) {
    out[i] = w2l::afToVector<float>(outBatched.col(i));
  }

  return std::make_pair(out, outstates);
}

AMUpdateFunc buildTransformerAmUpdateFunction(
    std::shared_ptr<SequenceCriterion>& c) {
  auto buf = std::make_shared<TS2SDecoderBuffer>(
      FLAGS_beamsize, FLAGS_attentionthreshold, FLAGS_smoothingtemperature);

  const TransformerCriterion* criterion =
      static_cast<TransformerCriterion*>(c.get());

  auto amUpdateFunc = [buf, criterion](
                          const float* emissions,
                          const int N,
                          const int T,
                          const std::vector<int>& rawY,
                          const std::vector<AMStatePtr>& rawPrevStates,
                          int& t) {
    if (t == 0) {
      buf->input = fl::Variable(af::array(N, T, emissions), false);
    }
    int B = rawY.size();
    buf->prevStates.resize(0);
    buf->ys.resize(0);

    for (int i = 0; i < B; i++) {
      TS2SState* prevState = static_cast<TS2SState*>(rawPrevStates[i].get());
      fl::Variable y;
      if (t > 0) {
        y = fl::constant(rawY[i], 1, s32, false);
      } else {
        prevState = &buf->dummyState;
      }
      buf->ys.push_back(y);
      buf->prevStates.push_back(prevState);
    }

    std::vector<std::vector<float>> amScores;
    std::vector<TS2SStatePtr> outStates;

    std::tie(amScores, outStates) = criterion->decodeBatchStep(
        buf->input,
        buf->ys,
        buf->prevStates,
        buf->attentionThreshold,
        buf->smoothingTemperature);

    std::vector<AMStatePtr> out;
    for (auto& os : outStates) {
      out.push_back(os);
    }

    return std::make_pair(amScores, out);
  };

  return amUpdateFunc;
}

std::string TransformerCriterion::prettyString() const {
  return "TransformerCriterion";
}

} // namespace w2l

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Seq2SeqCriterion.h"
#include <glog/logging.h>
#include <algorithm>
#include <queue>

using namespace fl;

namespace w2l {

Seq2SeqCriterion buildSeq2Seq(int numClasses, int eosIdx) {
  bool useSequentialDecoder = false;
  if ((FLAGS_pctteacherforcing < 100 &&
       FLAGS_samplingstrategy == w2l::kModelSampling) ||
      FLAGS_inputfeeding) {
    useSequentialDecoder = true;
  }

  std::shared_ptr<AttentionBase> attention;
  if (FLAGS_attention == w2l::kContentAttention) {
    attention = std::make_shared<ContentAttention>();
  } else if (FLAGS_attention == w2l::kKeyValueAttention) {
    attention = std::make_shared<ContentAttention>(true);
  } else if (FLAGS_attention == w2l::kNeuralContentAttention) {
    attention = std::make_shared<NeuralContentAttention>(FLAGS_encoderdim);
  } else if (FLAGS_attention == w2l::kSimpleLocationAttention) {
    useSequentialDecoder = true;
    attention = std::make_shared<SimpleLocationAttention>(FLAGS_attnconvkernel);
  } else if (FLAGS_attention == w2l::kLocationAttention) {
    useSequentialDecoder = true;
    attention = std::make_shared<LocationAttention>(
        FLAGS_encoderdim, FLAGS_attnconvkernel);
  } else if (FLAGS_attention == w2l::kNeuralLocationAttention) {
    useSequentialDecoder = true;
    attention = std::make_shared<NeuralLocationAttention>(
        FLAGS_encoderdim,
        FLAGS_attndim,
        FLAGS_attnconvchannel,
        FLAGS_attnconvkernel);
  } else {
    LOG(FATAL) << "unimplemented attention";
  }

  std::shared_ptr<WindowBase> window;
  if (FLAGS_attnWindow == w2l::kNoWindow) {
    window = nullptr;
  } else if (FLAGS_attnWindow == w2l::kMedianWindow) {
    if (FLAGS_trainWithWindow) {
      useSequentialDecoder = true;
    }
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
    LOG(FATAL) << "unimplemented window";
  }

  return Seq2SeqCriterion(
      numClasses,
      FLAGS_encoderdim,
      eosIdx,
      FLAGS_maxdecoderoutputlen,
      attention,
      window,
      FLAGS_trainWithWindow,
      FLAGS_pctteacherforcing,
      useSequentialDecoder,
      FLAGS_labelsmooth,
      FLAGS_inputfeeding,
      FLAGS_samplingstrategy);
}

Seq2SeqCriterion::Seq2SeqCriterion(
    int nClass,
    int hiddenDim,
    int eos,
    int maxDecoderOutputLen,
    std::shared_ptr<AttentionBase> attention,
    std::shared_ptr<WindowBase> window /* nullptr */,
    bool trainWithWindow /* false */,
    int pctTeacherForcing /* = 100 */,
    bool useSequentialDecoder /* = false */,
    double labelSmooth /* = 0.0 */,
    bool inputFeeding /* = false */,
    std::string samplingStrategy /* = w2l::kRandSampling */)
    : eos_(eos),
      maxDecoderOutputLen_(maxDecoderOutputLen),
      window_(window),
      trainWithWindow_(trainWithWindow),
      pctTeacherForcing_(pctTeacherForcing),
      useSequentialDecoder_(useSequentialDecoder),
      labelSmooth_(labelSmooth),
      inputFeeding_(inputFeeding),
      nClass_(nClass),
      samplingStrategy_(samplingStrategy) {
  add(std::make_shared<Embedding>(hiddenDim, nClass_));
  add(std::make_shared<RNN>(hiddenDim, hiddenDim, 1, RnnMode::GRU, false, 0.0));
  add(std::make_shared<Linear>(hiddenDim, nClass_));
  add(attention);
  params_.push_back(uniform(af::dim4{hiddenDim}, -1e-1, 1e-1));
}

std::vector<Variable> Seq2SeqCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("Invalid inputs size");
  }
  const auto& input = inputs[0];
  const auto& target = inputs[1];

  Variable out, alpha;
  if (useSequentialDecoder_) {
    std::tie(out, alpha) = decoder(input, target);
  } else {
    std::tie(out, alpha) = vectorizedDecoder(input, target);
  }

  // [nClass, targetlen, batchsize],
  out = logSoftmax(out, 0);

  auto losses = moddims(
      sum(categoricalCrossEntropy(out, target, ReduceMode::NONE), {0}), -1);
  if (train_ && labelSmooth_ > 0) {
    size_t nClass = out.dims(0);
    auto smoothLoss = moddims(sum(out, {0, 1}), -1);
    losses = (1 - labelSmooth_) * losses - (labelSmooth_ / nClass) * smoothLoss;
  }

  return {losses};
}

std::pair<Variable, Variable> Seq2SeqCriterion::vectorizedDecoder(
    const Variable& input,
    const Variable& target) {
  int U = target.dims(0);
  int B = target.dims(1);
  int T = input.dims(1);

  auto hy = tile(startEmbedding(), {1, 1, B});

  if (U > 1) {
    // Slice off eos
    auto y = target(af::seq(0, U - 2), af::span);
    if (train_) {
      if (samplingStrategy_ == w2l::kModelSampling) {
        throw std::logic_error(
            "vectorizedDecoder does not support model sampling");
      } else if (samplingStrategy_ == w2l::kRandSampling) {
        auto mask =
            Variable(af::randu(y.dims()) * 100 <= pctTeacherForcing_, false);
        auto samples =
            Variable((af::randu(y.dims()) * (nClass_ - 1)).as(s32), false);
        y = mask * y + (1 - mask) * samples;
      }
    }

    // [hiddendim, targetlen-1, batchsize]
    auto yEmbed = embedding()->forward(y);

    // [hiddendim, targetlen, batchsize]
    hy = concatenate({hy, yEmbed}, 1);
  }

  // transpose batch and target just for RNN
  hy = reorder(hy, 0, 2, 1);
  hy = decodeRNN()->forward(hy);
  hy = reorder(hy, 0, 2, 1);

  Variable windowWeight;
  if (window_ && (!train_ || trainWithWindow_)) {
    windowWeight = window_->computeWindowMask(U, T, B);
  }

  Variable alpha;
  Variable summaries;
  std::tie(alpha, summaries) = attention()->forward(
      hy,
      input,
      Variable(), // vectorizedDecoder does not support prev_attn input
      windowWeight);

  // [nClass, targetlen, batchsize]
  auto out = linearOut()->forward(summaries + hy);

  return std::make_pair(out, alpha);
}

std::pair<Variable, Variable> Seq2SeqCriterion::decoder(
    const Variable& input,
    const Variable& target) {
  int U = target.dims(0);

  if (window_) { // for softPretrainWindow
    window_->setBatchStat(input.dims(1), U, input.dims(2));
  }

  std::vector<Variable> outvec;
  std::vector<Variable> alphaVec;
  Seq2SeqState state;
  Variable y;
  for (int u = 0; u < U; u++) {
    Variable ox;
    std::tie(ox, state) = decodeStep(input, y, state);
    outvec.push_back(ox);
    alphaVec.push_back(state.alpha);
    if (!train_ ||
        af::allTrue<bool>(
            af::randu(1) * 100 <= af::constant(pctTeacherForcing_, 1))) {
      y = target(u, af::span);
    } else if (samplingStrategy_ == w2l::kModelSampling) {
      af::array maxIdx, maxValues;
      max(maxValues, maxIdx, ox.array());
      y = Variable(maxIdx, false);
    } else if (samplingStrategy_ == w2l::kRandSampling) {
      y = Variable(
          (af::randu(af::dim4{1, target.dims(1)}) * (nClass_ - 1)).as(s32),
          false);
    } else {
      throw std::invalid_argument("Invalid sampling strategy");
    }
  }

  // [nClass, targetlen, batchsize]
  auto out = concatenate(outvec, 1);

  // [targetlen, seqlen, batchsize]
  auto alpha = concatenate(alphaVec, 0);

  return std::make_pair(out, alpha);
}

af::array Seq2SeqCriterion::viterbiPath(const af::array& input) {
  return viterbiPathBase(input, false).first;
}

std::pair<af::array, Variable> Seq2SeqCriterion::viterbiPathBase(
    const af::array& input,
    bool saveAttn) {
  //  TODO: Extend to batchsize > 1
  bool wasTrain = train_;
  eval();
  std::vector<int> maxPath;
  std::vector<Variable> alphaVec;
  Variable alpha;
  Seq2SeqState state;
  Variable y, ox;
  af::array maxIdx, maxValues;
  int pred;
  for (int u = 0; u < maxDecoderOutputLen_; u++) {
    std::tie(ox, state) = decodeStep(Variable(input, false), y, state);
    max(maxValues, maxIdx, ox.array());
    maxIdx.host(&pred);
    if (saveAttn) {
      alphaVec.push_back(state.alpha);
    }

    if (pred == eos_) {
      break;
    }
    y = constant(pred, 1, s32, false);
    maxPath.push_back(pred);
  }
  if (saveAttn) {
    alpha = concatenate(alphaVec, 0);
  }

  if (wasTrain) {
    train();
  }
  af::array vPath =
      maxPath.empty() ? af::array() : af::array(maxPath.size(), maxPath.data());
  return std::make_pair(vPath, alpha);
}

std::vector<int> Seq2SeqCriterion::beamPath(
    const af::array& input,
    int beamSize /* = 10 */) {
  std::vector<Seq2SeqCriterion::CandidateHypo> beam;
  beam.emplace_back(CandidateHypo{});
  auto beamPaths = beamSearch(input, beam, beamSize, maxDecoderOutputLen_);
  return beamPaths[0].path;
}

// beam are candidates that need to be extended
std::vector<Seq2SeqCriterion::CandidateHypo> Seq2SeqCriterion::beamSearch(
    const af::array& input,
    std::vector<Seq2SeqCriterion::CandidateHypo> beam,
    int beamSize = 10,
    int maxLen = 200) {
  bool wasTrain = train_;
  eval();

  std::vector<Seq2SeqCriterion::CandidateHypo> complete;
  std::vector<Seq2SeqCriterion::CandidateHypo> newBeam;
  auto cmpfn = [](Seq2SeqCriterion::CandidateHypo& lhs,
                  Seq2SeqCriterion::CandidateHypo& rhs) {
    return lhs.score > rhs.score;
  };

  for (int l = 0; l < maxLen; l++) {
    newBeam.resize(0);
    for (auto& hypo : beam) {
      Variable y;
      if (!hypo.path.empty()) {
        y = constant(hypo.path.back(), 1, s32, false);
      }

      Variable ox;
      Seq2SeqState state;
      std::tie(ox, state) = decodeStep(Variable(input, false), y, hypo.state);
      ox = logSoftmax(ox, 0);
      auto oxVector = w2l::afToVector<float>(ox.array());
      for (int idx = 0; idx < oxVector.size(); idx++) {
        std::vector<int> path_(hypo.path);
        path_.push_back(idx);
        newBeam.emplace_back(hypo.score + oxVector[idx], path_, state);
      }
    }

    std::partial_sort(
        newBeam.begin(),
        newBeam.begin() +
            std::min(2 * beamSize, static_cast<int>(newBeam.size())),
        newBeam.end(),
        cmpfn);

    beam.resize(0);
    for (int idx = 0; idx < newBeam.size(); idx++) {
      auto& hypo = newBeam[idx];
      // We only move the top beamSize hypothesises into complete.
      if (idx < beamSize && hypo.path.back() == eos_) {
        hypo.path.pop_back();
        complete.push_back(hypo);
      } else if (hypo.path.back() != eos_) {
        beam.push_back(hypo);
      }
      if (beam.size() >= beamSize) {
        break;
      }
    }

    if (complete.size() >= beamSize) {
      std::partial_sort(
          complete.begin(), complete.begin() + beamSize, complete.end(), cmpfn);
      complete.resize(beamSize);

      // if lowest score in complete is better than best future hypo
      // then its not possible for any future hypothesis to replace existing
      // hypothesises in complete.
      if (complete.back().score > beam[0].score) {
        break;
      }
    }
  }

  if (wasTrain) {
    train();
  }

  return complete.empty() ? beam : complete;
}

std::pair<Variable, Seq2SeqState> Seq2SeqCriterion::decodeStep(
    const Variable& xEncoded,
    const Variable& y,
    const Seq2SeqState& inState) const {
  // xEncoded is shape [hiddendim, seqlen, batchsize]
  // y (if not empty) is shape [1, batchsize]
  size_t stepSize = af::getMemStepSize();
  af::setMemStepSize(10 * (1 << 10));
  Variable hy;
  if (y.isempty()) {
    hy = tile(startEmbedding(), {1, 1, static_cast<int>(xEncoded.dims(2))});
  } else {
    hy = embedding()->forward(y);
    if (inputFeeding_) {
      hy = hy + moddims(inState.summary, hy.dims());
    }
  }

  // [hiddendim, batchsize]
  hy = moddims(hy, {hy.dims(0), -1});

  Seq2SeqState outState;
  outState.step = inState.step + 1;
  std::tie(hy, outState.hidden) = decodeRNN()->forward(hy, inState.hidden);

  // [hiddendim, 1, batchsize]
  hy = moddims(hy, {hy.dims(0), 1, hy.dims(1)});

  Variable windowWeight;
  if (window_ && (!train_ || trainWithWindow_)) {
    windowWeight = window_->computeSingleStepWindow(
        inState.alpha, xEncoded.dims(1), xEncoded.dims(2), inState.step);
  }

  std::tie(outState.alpha, outState.summary) =
      attention()->forward(hy, xEncoded, inState.alpha, windowWeight);

  // [nClass, 1, batchsize]
  auto out = linearOut()->forward(outState.summary + hy);
  af::setMemStepSize(stepSize);
  return std::make_pair(out, outState);
}

std::pair<std::vector<std::vector<float>>, std::vector<Seq2SeqStatePtr>>
Seq2SeqCriterion::decodeBatchStep(
    const Variable& xEncoded,
    std::vector<Variable>& ys,
    const std::vector<Seq2SeqStatePtr>& inStates) const {
  // xEncoded is shape [hiddendim, seqlen, batchsize]
  // y (if not empty) is shape [1, batchsize]
  size_t stepSize = af::getMemStepSize();
  af::setMemStepSize(10 * (1 << 10));
  int batchSize = ys.size();
  std::vector<Variable> statesVector(batchSize);

  // Batch Ys
  for (int i = 0; i < batchSize; i++) {
    if (ys[i].isempty()) {
      ys[i] = startEmbedding();
    } else {
      ys[i] = embedding()->forward(ys[i]);
      if (inputFeeding_) {
        ys[i] = ys[i] + moddims(inStates[i]->summary, ys[i].dims());
      }
    }
    ys[i] = moddims(ys[i], {ys[i].dims(0), -1});
  }
  // yBatched [hiddendim, batchsize]
  Variable yBatched = concatenate(ys, 1);

  // Batch inState
  for (int i = 0; i < batchSize; i++) {
    statesVector[i] = inStates[i]->hidden;
  }
  Variable inStateHiddenBatched = concatenate(statesVector, 1);

  /* (1) RNN forward */
  Variable outStateBatched;
  std::tie(yBatched, outStateBatched) =
      decodeRNN()->forward(yBatched, inStateHiddenBatched);

  std::vector<Seq2SeqStatePtr> outstates(batchSize);
  for (int i = 0; i < batchSize; i++) {
    outstates[i] = std::make_shared<Seq2SeqState>();
    outstates[i]->step = inStates[i]->step + 1;
    outstates[i]->hidden = outStateBatched.col(i);
  }

  /* (2) Attention forward */
  if (window_ && (!train_ || trainWithWindow_)) {
    LOG(FATAL) << "Batched decoding does not support window models";
  } else {
    Variable alphaBatched;
    // DEBUG: Third Variable is set to empty since no attention use it.
    std::tie(alphaBatched, outStateBatched) =
        attention()->forward(yBatched, xEncoded, Variable(), Variable());
    // alphaBatched [T, 1, batchSize]
    alphaBatched = reorder(alphaBatched, 1, 2, 0);
    // outStateBatched [hidden_dim, 1, batchSize]
    outStateBatched = moddims(
        outStateBatched, {outStateBatched.dims(0), 1, outStateBatched.dims(1)});
    for (int i = 0; i < batchSize; i++) {
      outstates[i]->alpha = alphaBatched(af::span, af::span, i);
      outstates[i]->summary = outStateBatched(af::span, af::span, i);
    }
  }

  /* (3) Linear forward */
  // outBatched [nclass, 1, batchsize]
  yBatched = moddims(yBatched, {yBatched.dims(0), 1, yBatched.dims(1)});
  auto outBatched = linearOut()->forward(outStateBatched + yBatched);
  outBatched = logSoftmax(outBatched, 0);
  std::vector<std::vector<float>> out(batchSize);
  for (int i = 0; i < batchSize; i++) {
    out[i] = w2l::afToVector<float>(outBatched(af::span, af::span, i));
  }

  af::setMemStepSize(stepSize);
  return std::make_pair(out, outstates);
}

std::string Seq2SeqCriterion::prettyString() const {
  return "Seq2SeqCriterion";
}

} // namespace w2l

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/attention/LocationAttention.h"

using namespace fl;

namespace w2l {

SimpleLocationAttention::SimpleLocationAttention(int convKernel) {
  Sequential pa;
  pa.add(Conv2D(1, 1, 1, convKernel, 1, 1, -1, -1));
  pa.add(Reorder(2, 0, 1, 3));
  pa.add(ReLU());
  add(pa);
}

std::pair<Variable, Variable> SimpleLocationAttention::forward(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& prevAttn,
    const Variable& attnWeight) {
  int U = state.dims(1);
  if (U > 1) {
    throw std::invalid_argument(
        prettyString() + " only works on single step forward");
  }

  int T = xEncoded.dims(1);
  int B = xEncoded.dims(2);

  // [1, seqlen, batchsize]
  auto innerProd = matmulTN(state, xEncoded);

  if (!prevAttn.isempty()) {
    auto Ha = moddims(
        module(0)->forward({moddims(prevAttn, {1, T, 1, B})}).front(),
        {1, T, B});
    innerProd = innerProd + Ha;
  }

  if (!attnWeight.isempty()) {
    innerProd = innerProd + log(attnWeight);
  }

  // [1, seqlen, batchsize]
  auto attention = softmax(innerProd, 1);

  // [hiddendim, 1, batchsize]
  auto summaries = matmulNT(xEncoded, attention);

  return std::make_pair(attention, summaries);
}

std::string SimpleLocationAttention::prettyString() const {
  return "SimpleLocationBasedAttention";
}

LocationAttention::LocationAttention(int encDim, int convKernel) {
  Sequential pa;
  pa.add(Conv2D(1, encDim, 1, convKernel, 1, 1, -1, -1));
  pa.add(Reorder(2, 0, 1, 3));
  pa.add(ReLU());
  add(pa);
}

std::pair<Variable, Variable> LocationAttention::forward(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& prevAttn,
    const Variable& attnWeight) {
  int U = state.dims(1);
  if (U > 1) {
    throw std::invalid_argument(
        prettyString() + " only works on single step forward");
  }

  int H = xEncoded.dims(0);
  int T = xEncoded.dims(1);
  int B = xEncoded.dims(2);

  auto innerProd = matmulTN(state, xEncoded);

  if (!prevAttn.isempty()) {
    auto Ha = moddims(
        module(0)->forward({moddims(prevAttn, {1, T, 1, B})}).front(),
        {H, T, B});
    innerProd = innerProd + matmulTN(state, Ha);
  }

  if (!attnWeight.isempty()) {
    innerProd = innerProd + log(attnWeight);
  }

  // [1, seqlen, batchsize]
  auto attention = softmax(innerProd, 1);

  // [hiddendim, 1, batchsize]
  auto summaries = matmulNT(xEncoded, attention);

  return std::make_pair(attention, summaries);
}

std::string LocationAttention::prettyString() const {
  return "LocationBasedAttention";
}

NeuralLocationAttention::NeuralLocationAttention(
    int encDim,
    int attnDim,
    int convChannel,
    int convKernel) {
  add(Linear(encDim, attnDim));
  add(Linear(encDim, attnDim, false));
  Sequential pa;
  pa.add(Conv2D(1, convChannel, 1, convKernel, 1, 1, -1, -1));
  pa.add(Reorder(2, 0, 1, 3));
  pa.add(Linear(convChannel, attnDim, false));
  add(pa);
  add(Tanh());
  add(Linear(attnDim, 1, false));
}

std::pair<Variable, Variable> NeuralLocationAttention::forward(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& prevAttn,
    const Variable& attnWeight) {
  int U = state.dims(1);
  if (U > 1) {
    throw std::invalid_argument(
        prettyString() + " only works on single step forward");
  }

  int T = xEncoded.dims(1);
  int B = xEncoded.dims(2);

  auto Hx = module(0)->forward({xEncoded}).front();
  auto tileHy = tile(module(1)->forward({state}).front(), {1, T, 1});

  // [1, seqlen, batchsize]
  auto hidden = Hx + tileHy;
  if (!prevAttn.isempty()) {
    auto Ha = moddims(
        module(2)->forward({moddims(prevAttn, {1, T, 1, B})}).front(),
        {-1, T, B});
    hidden = hidden + Ha;
  }
  hidden = module(3)->forward({hidden}).front();
  auto nnOut = module(4)->forward({hidden}).front();

  if (!attnWeight.isempty()) {
    nnOut = nnOut + log(attnWeight);
  }

  // [1, seqlen, batchsize]
  auto attention = softmax(nnOut, 1);

  // [hiddendim, 1, batchsize]
  auto summaries = matmulNT(xEncoded, attention);

  return std::make_pair(attention, summaries);
}

std::string NeuralLocationAttention::prettyString() const {
  return "NeuralLocationBasedAttention";
}

} // namespace w2l

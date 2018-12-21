/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark Librispeech SOTA model on torch7 and Flashlight
 * Torch 7:
 * (https://github.com/soumith/convnet-benchmarks with `cudnn.fastest = true`)
 * Running on device: Tesla M40
 * ModelType: Librispeech      Kernels: cudnn  Input shape: 40x1x8000
 * cudnn                                   :updateOutput():     658.27
 * cudnn                                :updateGradInput():     842.65
 * cudnn                              :accGradParameters():    1057.63
 * cudnn                                          :Forward:     658.27
 * cudnn                                         :Backward:    1900.28
 * cudnn                                            :TOTAL:    2558.55
 *
 *
 * Flashlight:
 * ArrayFire v3.6.0 (CUDA, 64-bit Linux, build default)
 * Platform: CUDA Toolkit 8, Driver: 390.40
 * [0] Tesla M40, 11449 MB, CUDA Compute 5.2
 * -1- Tesla M40, 11449 MB, CUDA Compute 5.2
 * Total time (fwd+bwd pass) 2588.7 msec
 */

#include <flashlight/flashlight.h>

#include <iomanip>
#include <iostream>

#include <arrayfire.h>

using namespace fl;

Sequential model;
Variable input2, gradoutput, b;

static void fn2() {
  auto b = model.forward(input2);
  b.backward(gradoutput);
}

int main() {
  af::info();

  // https://github.com/facebookresearch/wav2letter/blob/master/arch/librispeech-glu-highdropout
  model.add(Conv2D(40, 400, 13, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.2));
  model.add(Conv2D(200, 440, 14, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.214));
  model.add(Conv2D(220, 484, 15, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.22898));
  model.add(Conv2D(242, 532, 16, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.2450086));
  model.add(Conv2D(266, 584, 17, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.262159202));
  model.add(Conv2D(292, 642, 18, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.28051034614));
  model.add(Conv2D(321, 706, 19, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.30014607037));
  model.add(Conv2D(353, 776, 20, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.321156295296));
  model.add(Conv2D(388, 852, 21, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.343637235966));
  model.add(Conv2D(426, 936, 22, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.367691842484));
  model.add(Conv2D(468, 1028, 23, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.393430271458));
  model.add(Conv2D(514, 1130, 24, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.42097039046));
  model.add(Conv2D(565, 1242, 25, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.450438317792));
  model.add(Conv2D(621, 1366, 26, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.481969000038));
  model.add(Conv2D(683, 1502, 27, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.51570683004));
  model.add(Conv2D(751, 1652, 28, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.551806308143));
  model.add(Conv2D(826, 1816, 29, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.590432749713));
  model.add(Conv2D(908, 1816, 1, 1));
  model.add(GatedLinearUnit(2));
  model.add(Dropout(0.590432749713));
  model.add(Conv2D(908, 30, 1, 1));

  input2 = Variable(af::randu(8000, 1, 40, 1) * 2 - 1, false); // ~ 8 sec audio

  int ntimes = 25;

  b = model.forward(input2);
  gradoutput = Variable(af::randu(b.dims()) * 2 - 2, false);
  for (int i = 0; i < 5; ++i) {
    fn2(); // warmup
  }

  af::sync();
  auto s = af::timer::start();
  for (int i = 0; i < ntimes; ++i) {
    fn2();
  }
  af::sync();
  auto e = af::timer::stop(s);
  std::cout << "Total time (fwd+bwd pass) " << std::setprecision(5)
            << e * 1000.0 / ntimes << " msec" << std::endl;
  return 0;
}

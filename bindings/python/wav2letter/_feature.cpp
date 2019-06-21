/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libraries/feature/Ceplifter.h"
#include "libraries/feature/Dct.h"
#include "libraries/feature/Derivatives.h"
#include "libraries/feature/Dither.h"
#include "libraries/feature/FeatureParams.h"
#include "libraries/feature/Mfcc.h"
#include "libraries/feature/Mfsc.h"
#include "libraries/feature/PowerSpectrum.h"
#include "libraries/feature/PreEmphasis.h"
#include "libraries/feature/SpeechUtils.h"
#include "libraries/feature/TriFilterbank.h"
#include "libraries/feature/Windowing.h"

namespace py = pybind11;
using namespace pybind11::literals;

using WindowType = w2l::WindowType;
using FrequencyScale = w2l::FrequencyScale;
using FeatureParams = w2l::FeatureParams;

using Ceplifter = w2l::Ceplifter<float>;
using Dct = w2l::Dct<float>;
using Derivatives = w2l::Derivatives<float>;
using Dither = w2l::Dither<float>;
using Mfcc = w2l::Mfcc<float>;
using Mfsc = w2l::Mfsc<float>;
using PowerSpectrum = w2l::PowerSpectrum<float>;
using PreEmphasis = w2l::PreEmphasis<float>;
using TriFilterbank = w2l::TriFilterbank<float>;
using Windowing = w2l::Windowing<float>;

PYBIND11_MODULE(_feature, m) {
  py::enum_<WindowType>(m, "WindowType")
      .value("HAMMING", WindowType::HAMMING)
      .value("HANNING", WindowType::HANNING);
  py::enum_<FrequencyScale>(m, "FrequencyScale")
      .value("MEL", FrequencyScale::MEL)
      .value("LINEAR", FrequencyScale::LINEAR)
      .value("LOG10", FrequencyScale::LOG10);
  py::class_<FeatureParams>(m, "FeatureParams")
      .def(
          py::init<
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              WindowType,
              float,
              float,
              float,
              bool,
              bool,
              bool,
              bool>(),
          "sampling_freq"_a = 16000,
          "frame_size_ms"_a = 25,
          "frame_stride_ms"_a = 10,
          "num_filterbank_chans"_a = 23,
          "low_freq_filterbank"_a = 0,
          "high_freq_filterbank"_a = -1,
          "num_cepstral_coeffs"_a = 13,
          "lifter_param"_a = 22,
          "delta_window"_a = 2,
          "acc_window"_a = 2,
          "window_type"_a = WindowType::HAMMING,
          "preem_coef"_a = 0.97,
          "mel_floor"_a = 1.0,
          "dither_val"_a = 0.0,
          "use_power"_a = true,
          "use_energy"_a = true,
          "raw_energy"_a = true,
          "zero_mean_frame"_a = true)
      .def("numFrameSizeSamples", &FeatureParams::numFrameSizeSamples)
      .def("numFrameStrideSamples", &FeatureParams::numFrameStrideSamples)
      .def("nFft", &FeatureParams::nFft)
      .def("filterFreqResponseLen", &FeatureParams::filterFreqResponseLen)
      .def("powSpecFeatSz", &FeatureParams::powSpecFeatSz)
      .def("mfscFeatSz", &FeatureParams::mfscFeatSz)
      .def("mfccFeatSz", &FeatureParams::mfccFeatSz)
      .def("numFrames", &FeatureParams::numFrames)
      .def_readwrite("samplingFreq", &FeatureParams::samplingFreq)
      .def_readwrite("frameSizeMs", &FeatureParams::frameSizeMs)
      .def_readwrite("frameStrideMs", &FeatureParams::frameStrideMs)
      .def_readwrite("numFilterbankChans", &FeatureParams::numFilterbankChans)
      .def_readwrite("lowFreqFilterbank", &FeatureParams::lowFreqFilterbank)
      .def_readwrite("highFreqFilterbank", &FeatureParams::highFreqFilterbank)
      .def_readwrite("numCepstralCoeffs", &FeatureParams::numCepstralCoeffs)
      .def_readwrite("lifterParam", &FeatureParams::lifterParam)
      .def_readwrite("deltaWindow", &FeatureParams::deltaWindow)
      .def_readwrite("accWindow", &FeatureParams::accWindow)
      .def_readwrite("windowType", &FeatureParams::windowType)
      .def_readwrite("preemCoef", &FeatureParams::preemCoef)
      .def_readwrite("melFloor", &FeatureParams::melFloor)
      .def_readwrite("ditherVal", &FeatureParams::ditherVal)
      .def_readwrite("usePower", &FeatureParams::usePower)
      .def_readwrite("useEnergy", &FeatureParams::useEnergy)
      .def_readwrite("rawEnergy", &FeatureParams::rawEnergy)
      .def_readwrite("zeroMeanFrame", &FeatureParams::zeroMeanFrame);

  py::class_<Ceplifter>(m, "Ceplifter")
      .def(py::init<int64_t, int64_t>())
      .def("apply", &Ceplifter::apply)
      .def("applyInPlace", &Ceplifter::applyInPlace);
  py::class_<Dct>(m, "Dct")
      .def(py::init<int64_t, int64_t>())
      .def("apply", &Dct::apply);
  py::class_<Derivatives>(m, "Derivatives")
      .def(py::init<int64_t, int64_t>())
      .def("apply", &Derivatives::apply);
  py::class_<Dither>(m, "Dither")
      .def(py::init<float>())
      .def("apply", &Dither::apply)
      .def("applyInPlace", &Dither::applyInPlace);
  py::class_<Mfcc>(m, "Mfcc")
      .def(py::init<const FeatureParams&>())
      .def("apply", &Mfcc::apply)
      .def("batchApply", &Mfcc::batchApply)
      .def("outputSize", &Mfcc::outputSize)
      .def("getFeatureParams", &Mfcc::getFeatureParams);
  py::class_<Mfsc>(m, "Mfsc")
      .def(py::init<const FeatureParams&>())
      .def("apply", &Mfsc::apply)
      .def("batchApply", &Mfsc::batchApply)
      .def("outputSize", &Mfsc::outputSize)
      .def("getFeatureParams", &Mfsc::getFeatureParams);
  py::class_<PowerSpectrum>(m, "PowerSpectrum")
      .def(py::init<const FeatureParams&>())
      .def("apply", &PowerSpectrum::apply)
      .def("batchApply", &PowerSpectrum::batchApply)
      .def("outputSize", &PowerSpectrum::outputSize)
      .def("getFeatureParams", &PowerSpectrum::getFeatureParams);
  py::class_<PreEmphasis>(m, "PreEmphasis")
      .def(py::init<float, int64_t>())
      .def("apply", &PreEmphasis::apply)
      .def("applyInPlace", &PreEmphasis::applyInPlace);
  py::class_<TriFilterbank>(m, "TriFilterbank")
      .def(py::init<
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           FrequencyScale>())
      .def("apply", &TriFilterbank::apply)
      .def("filterbank", &TriFilterbank::filterbank);
  py::class_<Windowing>(m, "Windowing")
      .def(py::init<int64_t, WindowType>())
      .def("apply", &Windowing::apply)
      .def("applyInPlace", &Windowing::applyInPlace);

  m.def("frameSignal", w2l::frameSignal<float>);
  m.def("cblasGemm", w2l::cblasGemm<float>);
}

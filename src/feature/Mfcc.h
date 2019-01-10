/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Ceplifter.h"
#include "Dct.h"
#include "Derivatives.h"
#include "FeatureParams.h"
#include "Mfsc.h"
#include "SpeechUtils.h"

namespace speech {

// Computes Mel Frequency Cepstral Coefficient (MFCC) for a speech signal.
// Feature calculation is similar to the calculation of default HTK MFCCs except
// for Energy Normalization.
// TODO: Support
//  1. ENORMALIZE
//  2. Cepstral Mean Normalisation
//  3. Vocal Tract Length Normalisation
// Example usage:
//   FeatureParams params;
//   Mfcc mfcc(params); af::array input = af::randu(123456, 987);
//   af::array mfccfeatures = mfcc->apply(input);
//
// References
//  [1] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D.,
//     Liu, X., Moore, G., Odell, J., Ollason, D., Povey, D.,
//      Valtchev, V., Woodland, P., 2006. The HTK Book (for HTK
//      Version 3.4.1). Engineering Department, Cambridge University.
//      URL: http://htk.eng.cam.ac.uk
//  [2] Daniel Povey , Arnab Ghoshal , Gilles Boulianne , Nagendra Goel , Mirko
//      Hannemann , Yanmin Qian , Petr Schwarz , Georg Stemmer - The kaldi
//      speech recognition toolkit
//      URL: http://kaldi-asr.org/
//  [3] Ellis, D., 2005. PLP and RASTA (and MFCC, and inversion) in Matlab
//      URL: https://labrosa.ee.columbia.edu/matlab/rastamat/
//  [4] Huang, X., Acero, A., Hon, H., 2001. Spoken Language
//      Processing: A guide to theory, algorithm, and system
//      development. Prentice Hall, Upper Saddle River, NJ,
//      USA (pp. 314-315).
//  [5] Kamil Wojcicki, HTK MFCC MATLAB,
//      URL:
//      https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab
//

template <typename T>
class Mfcc : public Mfsc<T> {
 public:
  explicit Mfcc(const FeatureParams& params);

  virtual ~Mfcc() {}

  // input - input speech signal (T)
  // Returns - MFCC features (Col Major : FEAT X FRAMESZ)
  std::vector<T> apply(const std::vector<T>& input) override;

  int64_t outputSize(int64_t inputSz) override;

 private:
  // The following classes are defined in the order they are applied
  Dct<T> dct_;
  Ceplifter<T> ceplifter_;
  Derivatives<T> derivatives_;

  void validateMfccParams() const;
};
} // namespace speech

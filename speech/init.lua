-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('torch')
local speech = require('libspeech')

--Each of these functions may contain library functions, that do not
--parallelize high dimensional (batched) input, so the parallelization
--happens within sequential or other wrapper pipelines.
--The functions that these functionals return should not do little
--allocations and be as small and efficient as possible as they will
--be called repeatedly. This also means, that upon small changes to
--a preprocessing function's parameters, the function will have to
--be recreated using the functionals.

require('speech.proc')
require('speech.trifiltering')
require('speech.dct')
require('speech.fftw')
require('speech.preemphasis')
require('speech.windowing')
require('speech.ceplifter')
require('speech.mfcc')
require('speech.pow')
require('speech.mfsc')
require('speech.derivatives')

return speech

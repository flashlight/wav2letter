-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local speech = require('libspeech')

--Applies simple 1d fft, yields complex result
local function Fftw_1d_fft()
   return function (output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)
      output:resize(input:size(1), 2)
      output:zero()
      input.speech.Fftw_forward(input, output)
      return output
   end
end
--TODO: Separate plan function call (return function with steady plan)
--TODO: Option for optimal plan creation
--TODO: Use rank n function

speech.Fftw_1d_fft = Fftw_1d_fft

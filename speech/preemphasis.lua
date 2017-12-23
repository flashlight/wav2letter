-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local speech = require('libspeech')

local function PreEmphasis(alpha)
   alpha = alpha or 0.97
   return function(output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)

      output:resize(input:size(1)):copy(input)
      output[{{2, input:size(1)}}]:add(-alpha,
                                   input[{{1, input:size(1)-1}}])
      return output
   end
end

speech.PreEmphasis = PreEmphasis

-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local speech = require('libspeech')

local function Proc(output_raw, input_raw)
   local input
   local output
   if not input_raw then
      input  = output_raw
      output = torch.Tensor():type(input:type())
   else
      input  = input_raw
      output = output_raw
   end

   return input, output
end

speech.Proc = Proc

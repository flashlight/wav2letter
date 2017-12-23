-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local speech = require('libspeech')

--dctm = @( N, M )( sqrt(2.0/M) * cos( repmat([0:N-1].',1,M) ...
--                                       .* repmat(pi*([1:M]-0.5)/M,N,1) ) );
local function Dct(nof, nocep)
   local dctm = torch.zeros(nocep, nof)
   for i = 1, nof do
      dctm[{{}, i}] = torch.range(0, nocep - 1)
   end
   for i = 1, nocep do
      dctm[{i, {}}]:cmul(torch.range(1, nof):add(-0.5):mul(math.pi):div(nof))
   end
   dctm:cos()
   dctm:mul(math.sqrt(2  / nof))

   return function (output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)
      dctm  = dctm:type(input:type())
      if (input:dim() == 1) then
         output:resize(nocep)
         output:zero()
         return output:addmv(dctm, input)
      elseif (input:dim() == 2) then
         output:resize(input:size(1), nocep)
         output:zero()
         return output:t():addmm(dctm, input:t())
      else
         error("Unsupported dimension")
      end
   end
end

speech.Dct = Dct

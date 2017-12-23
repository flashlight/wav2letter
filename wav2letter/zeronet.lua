-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ZeroNet, Module = torch.class('nn.ZeroNet', 'nn.Module')

function ZeroNet:__init(kw, dw, nclass)
   Module.__init(self)
   self.kw = kw
   self.dw = dw
   self.nclass = nclass
end


function ZeroNet:parameters()
    return {}, {}
end

function ZeroNet:updateOutput(input)
   local T = input:size(1)
   T = math.floor((T-self.kw)/self.dw)+1
   self.output:resize(T, self.nclass):zero()
   return self.output
end

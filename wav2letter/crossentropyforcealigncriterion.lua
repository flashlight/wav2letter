-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local CrossEntropyForceAlignCriterion, Criterion = torch.class('nn.CrossEntropyForceAlignCriterion', 'nn.Criterion')

function CrossEntropyForceAlignCriterion:__init(N, ismax, scale)
   scale = scale or function() return -1 end
   Criterion.__init(self)
   self.lsm = nn.LogSoftMax()
   self.fal = nn.ForceAlignCriterion(N, ismax, function(input, target) return -scale(input, target) end)
   self.transitions = self.fal.transitions
end

function CrossEntropyForceAlignCriterion:updateOutput(input, target)
   self.lsm:updateOutput(input)
   self.output = self.fal:updateOutput(self.lsm.output, target)
   return self.output
end

function CrossEntropyForceAlignCriterion:updateGradInput(input, target)
   local g = self.fal:updateGradInput(self.lsm.output, target)
   self.gradInput = self.lsm:updateGradInput(input, g)
   return self.gradInput
end

function CrossEntropyForceAlignCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

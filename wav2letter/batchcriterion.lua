-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local BatchCriterion = torch.class('nn.BatchCriterion', 'nn.Criterion')

--Share elements of table shares with each member of modules
--by default.
function BatchCriterion:__init(batchsize, shares, class, ...)
   self.gradInput = torch.Tensor()
   self.batchsize = batchsize
   local args = {...}
   local crit = nn[class](unpack(args))
   self.modules = {}
   self.modules[1] = crit
   for i = 2, batchsize do
      self.modules[i] = self.modules[1]:clone():share(self.modules[1], unpack(shares))
   end
   --Keep reference for semantics (e.g. :share(crit, 'transitions') in train.lua)
   for i, v in pairs(shares) do
      if self.modules[1][v] ~= nil then
         self[v] = self.modules[1][v]
      end
   end
end

function BatchCriterion:updateOutput(input, target)
   self.output = {}
   for i = 1, input:size(1) do
      self.output[i] = self.modules[i]:updateOutput(input[i], target[i])
   end
   return self.output
end

function BatchCriterion:viterbi(input)
   self.paths = {}
   self.scores = {}
   for i = 1, input:size(1) do
      local path, score = self.modules[i]:viterbi(input[i])
      self.paths[i] = path
      self.scores[i] = score
   end
   return self.paths, self.scores
end

function BatchCriterion:zeroGradParameters()
   for i = 1, #self.modules do
      if self.modules[i].zeroGradParameters then
         self.modules[i]:zeroGradParameters()
      end
   end
end

function BatchCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   for i = 1, input:size(1) do
      self.gradInput[i] = self.modules[i]:updateGradInput(input[i], target[i])
   end
   return self.gradInput
end

function BatchCriterion:updateParameters(lr)
   for i = 1, #self.modules do
      if self.modules[i].updateParameters then
         self.modules[i]:updateParameters(lr)
      end
   end
end

--Share with each member
function BatchCriterion:share(layer, ...)
   local arg = {...}
   for i = 1, self.batchsize do
      self.modules[i]:share(layer, ...)
   end
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

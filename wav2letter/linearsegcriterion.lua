-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local LinearSegCriterion = torch.class('nn.LinearSegCriterion', 'nn.Criterion')

function LinearSegCriterion:__init(N, ismax, scale, notrans, noC)
   self.notrans = notrans
   if notrans then
      print('| LinearSegCriterion uses CrossEntropyCriterion')
      self.fcc = nn.CrossEntropyCriterion()
   else
      if noC then
         print('| LinearSegCriterion uses FullConnectCriterion')
         self.fcc = nn.FullConnectCriterion(N, ismax, scale)
      else
         print('| LinearSegCriterion uses FullConnectCriterion (C based)')
         self.fcc = nn.FullConnectCriterionC(N, ismax, scale)
      end
   end
   self.lintarget = torch.LongTensor()
   self.transitions = self.fcc.transitions
end

function LinearSegCriterion:updateOutput(input, target)
   local T = input:size(1)
   local N = target:size(1)
   assert(N <= T, 'not enough frames to handle all labels')
   local lintarget = self.lintarget
   lintarget:resize(T)
   local alpha = (N-1)/(T-1)
   local beta = 1-alpha
   for t=1,T do
      lintarget[t] = target[alpha*t+beta]
   end
   self.output = self.fcc:updateOutput(input, lintarget)
   return self.output
end

function LinearSegCriterion:zeroGradParameters()
   if not self.notrans then
      self.fcc:zeroGradParameters()
   end
end

function LinearSegCriterion:updateGradInput(input, target)
   self.gradInput = self.fcc:updateGradInput(input, self.lintarget)
   return self.gradInput
end

function LinearSegCriterion:updateParameters(lr)
   if not self.notrans then
      self.fcc:updateParameters(lr)
   end
end

function LinearSegCriterion:parameters()
   if not self.notrans then
      return {self.fcc.transitions}, {self.fcc.gtransitions}
   end
end

function LinearSegCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self.fcc[v] ~= nil then
         self.fcc[v]:set(layer[v])
      end
   end
   return self
end

-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local utils = require 'wav2letter.utils'
local C = utils.C

local FullConnectCriterionC = torch.class('nn.FullConnectCriterionC', 'nn.Criterion')

function FullConnectCriterionC:__init(N, ismax, scale)
   ismax = ismax or false
   assert(ismax == false, "FullConnectCriterion (C based) does not support ismax")

   self.scale = scale or function(input, target) return 1 end
   self.T = 0
   self.N = N
   self.transitions = torch.zeros(N, N)
   self.gtransitions = torch.zeros(N, N)

   self.gradInput = torch.Tensor()
   self.macc      = torch.Tensor():long()
   self.acc       = torch.Tensor():double()
   self.gacc      = torch.Tensor():double()
end

function FullConnectCriterionC:realloc(T)
   self.T = math.max(T, self.T)
   print(string.format('[FullConnectCriterion (C based): reallocating with T=%d N=%d]', self.T, self.N))

   self.macc:resize(self.T, self.N)
   self.acc:resize(self.T, self.N)
   self.gacc:resize(self.T, self.N)
end

function FullConnectCriterionC:updateOutput(input, target)
   local T = input:size(1)
   local N = self.N
   local scale = self.scale(input, target)
   if T > self.T then
      self:realloc(T)
   end
   self.acc:zero()
   self.macc:zero()

   self.output = C.fccfw(input:cdata(),
                         self.transitions:cdata(),
                         self.macc:cdata(),
                         self.acc:cdata(),
                         T,
                         N)
   self.output = self.output * scale
   if target then
      assert(target:size(1) == T, 'input and target do not match')
      self.output = self.output + self:forwardTarget(input, target, scale)
   end
   return self.output
end

function FullConnectCriterionC:viterbi(input)
   error("FullConnectCriterion (C based) does not support viterbi")
end

function FullConnectCriterionC:zeroGradParameters()
   self.gtransitions:zero()
end

function FullConnectCriterionC:updateGradInput(input, target)
   local T = input:size(1)
   local N = self.N
   local scale = self.scale(input, target)
   self.gacc:zero()
   self.gradInput:resize(T, N)
   self.gradInput:zero()
   if target then
      self:backwardTarget(input, target, scale)
   end
   C.fccbw(input:cdata(),
           self.transitions:cdata(),
           self.gradInput:cdata(),
           self.gtransitions:cdata(),
           self.macc:cdata(),
           self.acc:cdata(),
           self.gacc:cdata(),
           scale,
           T,
           N)
   self.gradInput = self.gradInput:narrow(1, 1, T)
   return self.gradInput
end

function FullConnectCriterionC:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

-- could be in C
function FullConnectCriterionC:forwardTarget(input, target, scale)
   local T = input:size(1)
   local N = self.N
   local loss = 0
   local input_p = input:data()
   local target_p = target:data()
   local transitions_p = self.transitions:data()
   local lblm1
   for t=0,T-1 do
      local lbl = target_p[t]-1
      if lbl < 0 or lbl >= N then
         error('invalid target')
      end
      loss = loss - input_p[t*N+lbl]
      if t > 0 then
         loss = loss - transitions_p[lbl*N+lblm1]
      end
      lblm1 = lbl
   end
   return loss*scale
end

-- could be in C
function FullConnectCriterionC:backwardTarget(input, target, scale)
   local T = input:size(1)
   local N = self.N
   local gemissions_p = self.gradInput:data()
   local target_p = target:data()
   local gtransitions_p = self.gtransitions:data()
   local lblm1
   for t=0,T-1 do
      local lbl = target_p[t]-1
      gemissions_p[t*N+lbl] = gemissions_p[t*N+lbl] - scale
      if t > 0 then
         gtransitions_p[lbl*N+lblm1] = gtransitions_p[lbl*N+lblm1] - scale
      end
      lblm1 = lbl
   end
end

function FullConnectCriterionC:parameters()
   return {self.transitions}, {self.gtransitions}
end

function FullConnectCriterionC:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

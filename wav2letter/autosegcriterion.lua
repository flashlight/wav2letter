-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local AutoSegCriterion = torch.class('nn.AutoSegCriterion', 'nn.Criterion')

function AutoSegCriterion:__init(N, isposmax, isnegmax, scale, S, noC)
   noC = noC or false
   scale = scale or function(input, target) return 1 end
   local function falscale(input, target)
      self.__scale = scale(input, target)
      return -self.__scale
   end
   local function fccscale()
      return self.__scale
   end
   if S == 'garbage' then
      print('| AutoSegCriterion uses ForceAlignGarbageCriterion')
      self.fal = nn.ForceAlignGarbageCriterion(N, isposmax, falscale)
   else
      if isposmax or noC then
         print('| AutoSegCriterion uses ForceAlignCriterion')
         self.fal = nn.ForceAlignCriterion(N, isposmax, falscale)
      else
         print('| AutoSegCriterion uses ForceAlignCriterion (C based)')
         self.fal = nn.ForceAlignCriterionC(N, isposmax, falscale)
      end
   end
   if tonumber(S) then
      print('| AutoSegCriterion uses MultiStateFullConnectCriterion')
      self.fcc = nn.MultiStateFullConnectCriterion(N/S, S, isnegmax, fccscale)
   elseif S == 'garbage' then
      print('| AutoSegCriterion uses FullConnectGarbageCriterion')
      self.fcc = nn.FullConnectGarbageCriterion(N, isnegmax, fccscale)
   else
      if isnegmax or noC then
         print('| AutoSegCriterion uses FullConnectCriterion')
         self.fcc = nn.FullConnectCriterion(N, isnegmax, fccscale)
      else
         print('| AutoSegCriterion uses FullConnectCriterion (C based)')
         self.fcc = nn.FullConnectCriterionC(N, isnegmax, fccscale)
      end
   end
   self.fcc.transitions = self.fal.transitions
   self.transitions = self.fal.transitions
   self.gtransitions = torch.Tensor(N, N)
   self.gradInput = torch.Tensor()
end

function AutoSegCriterion:updateOutput(input, target)
   -- fal first because of scale(input, target)
   self.output = self.fal:updateOutput(input, target) + self.fcc:updateOutput(input)
   return self.output
end

function AutoSegCriterion:viterbi(input)
   return self.fcc:viterbi(input)
end

function AutoSegCriterion:zeroGradParameters()
   self.fcc:zeroGradParameters()
   self.fal:zeroGradParameters()
end

function AutoSegCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.fal:updateGradInput(input, target)
   self.fcc:updateGradInput(input)
   self.gradInput:copy(self.fal.gradInput)
   self.gradInput:add(self.fcc.gradInput)
   self.gtransitions:copy(self.fal.gtransitions)
   self.gtransitions:add(self.fcc.gtransitions)
   return self.gradInput
end

function AutoSegCriterion:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

function AutoSegCriterion:parameters()
   return {self.transitions}, {self.gtransitions}
end

function AutoSegCriterion:share(layer, ...)
   self.fcc:share(layer, ...)
   self.fal:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

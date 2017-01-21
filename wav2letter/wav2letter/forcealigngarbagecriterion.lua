local gtn = require 'gtn'

local ForceAlignGarbageCriterion, ForceAlignCriterion = torch.class('nn.ForceAlignGarbageCriterion', 'nn.ForceAlignCriterion')

function ForceAlignGarbageCriterion:__init(N, ismax, scale)
   ForceAlignCriterion.__init(self, N+1, ismax, scale)
   self.transitions = torch.zeros(N, N)
   self.gtransitions = torch.zeros(N, N)
   self.targetbuffer = torch.LongTensor()
end

function ForceAlignGarbageCriterion:insertGarbage(target)
   self.targetbuffer:resize(target:size(1)*2):fill(self.NL)
   self.targetbuffer:unfold(1, 1, 2):copy(target)
   return self.targetbuffer
end

function ForceAlignGarbageCriterion:updateOutput(input, target)
   target = self:insertGarbage(target)
   return ForceAlignCriterion.updateOutput(self, input, target)
end

-- could be in C
function ForceAlignGarbageCriterion:forwardParams(input, target)
   local T = input:size(1)
   local N = target:size(1)
   local NL = self.NL
   assert(input:size(2) == self.NL)
   local emissions = self.emissions:narrow(1, 1, T):narrow(2, 1, N)
   local subtransitions = self.subtransitions:narrow(1, 1, N)
   local transitions = self.transitions
   local idxm1
   for i=1,N do
      local idx = target[i]
      emissions:select(2, i):copy(input:select(2, idx))
      if idx == NL then
         idx = target[i-1]
      end
      subtransitions[i][1] = transitions[idx][idx]
      if idxm1 then
         subtransitions[i][2] = transitions[idx][idxm1]
      end
      idxm1 = idx
   end
end

function ForceAlignGarbageCriterion:updateGradInput(input, target)
   target = self:insertGarbage(target)
   return ForceAlignCriterion.updateGradInput(self, input, target)
end

-- could be in C
function ForceAlignGarbageCriterion:backwardParams(input, target)
   local T = input:size(1)
   local N = target:size(1)
   local NL = self.NL
   assert(input:size(2) == self.NL)
   local ginput = self.gradInput:resize(T, NL):zero()
   local gemissions = self.gemissions:narrow(1, 1, T)
   local gsubtransitions = self.gsubtransitions:narrow(1, 1, N)
   local gtransitions = self.gtransitions
   local idxm1
   for i=1,N do
      local idx = target[i]
      ginput:select(2, idx):add(gemissions:select(2, i))
      if idx == NL then
         idx = target[i-1]
      end
      gtransitions[idx][idx] = gtransitions[idx][idx] + gsubtransitions[i][1]
      if idxm1 then
         gtransitions[idx][idxm1] = gtransitions[idx][idxm1] + gsubtransitions[i][2]
      end
      idxm1 = idx
   end
end

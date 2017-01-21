local utils = require 'wav2letter.utils'
local C = utils.C

local ForceAlignCriterionC = torch.class('nn.ForceAlignCriterionC', 'nn.Criterion')

function ForceAlignCriterionC:__init(N, ismax, scale, S)
   S = S or 1
   assert(S == 1 , "ForceAlignCriterion (C based) does not support multistate")
   assert(ismax == false, "ForceAlignCriterion (C based) does not support ismax")

   self.scale = scale or function(input, target) return 1 end
   self.T = 0
   self.TN = 0
   self.N = N
   self.transitions = torch.zeros(N, N):float()
   self.gtransitions = torch.zeros(N, N):float()
   self.gsubtransitions = torch.zeros(N, N):float()

   self.gradInput = torch.Tensor()
   self.macc      = torch.Tensor():long()
   self.acc       = torch.Tensor():double()
   self.gacc      = torch.Tensor():double()
end

function ForceAlignCriterionC:realloc(T, TN)
   self.T = math.max(T, self.T)
   self.TN = math.max(TN, self.TN)

   print(string.format('[ForceAlignCriterion (C based): reallocating with T=%d N=%d TN=%d]', self.T, self.N, self.TN))

   self.macc:resize(self.T, self.N)
   self.acc:resize(self.T, self.TN)
   self.gacc:resize(self.T, self.TN)
end

function ForceAlignCriterionC:updateOutput(input, target)
   local T = input:size(1)
   local N = self.N
   local TN = target:size(1)
   local scale = self.scale(input, target)
   if T > self.T or TN > self.TN then
      self:realloc(T, TN)
   end
   self.gacc:zero()
   self.acc:zero()
   self.macc:zero()

   self.output = C.falfw(input:cdata(),
                         target:cdata(),
                         self.transitions:cdata(),
                         self.acc:cdata(),
                         self.macc:cdata(),
                         T,
                         N,
                         TN)
   self.output = self.output * scale
   return self.output
end

function ForceAlignCriterionC:viterbi(input)
   error("ForceAlignCriterion (C based) does not support viterbi")
end

function ForceAlignCriterionC:zeroGradParameters()
   self.gtransitions:zero()
end

function ForceAlignCriterionC:updateGradInput(input, target)
   local T = input:size(1)
   local N = self.N
   local TN = target:size(1)
   local scale = self.scale(input, target)
   self.gradInput:resize(T, N)
   self.gradInput:zero()
   C.falbw(input:cdata(),
             target:cdata(),
             self.transitions:cdata(),
             self.gradInput:cdata(),
             self.gtransitions:cdata(),
             self.acc:cdata(),
             self.gacc:cdata(),
             scale,
             T,
             N,
             TN)
   self.gradInput = self.gradInput:narrow(1, 1, T)
   return self.gradInput
end

function ForceAlignCriterionC:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

function ForceAlignCriterionC:parameters()
   return {self.transitions}, {self.gtransitions}
end

function ForceAlignCriterionC:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

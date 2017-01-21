local ffi = require 'ffi'
local utils = require 'wav2letter.utils'
local C = utils.C

local BatchAutoSegCriterionC = torch.class('nn.BatchAutoSegCriterionC',
                                            'nn.Criterion')

function BatchAutoSegCriterionC:__init(B, N, isposmax, isnegmax, scale, S)
   S = S or 1
   isposmax = isposmax or false
   isnegmax = isnegmax or false
   assert(S == 1 , "AutoSegCriterion (C based) does not support multistate")
   assert(isposmax == false, "AutoSegCriterion (C based) does not support " ..
                             "isposmax")
   assert(isnegmax == false, "AutoSegCriterion (C based) does not support " ..
                             "isnegmax")

   self.B = B
   self.scale = scale or function(input, target) return 1 end
   self.T = 0
   self.Tt = torch.zeros(B):long()
   self.TN = 0
   self.TNt = torch.zeros(B):long()
   self.N = N
   self.transitions = torch.zeros(N, N)
   self.gtransitions = torch.zeros(B, N, N)
   self.fccgtransitions = torch.zeros(B, N, N)

   self.gradInput = {}
   self.gradInputT = torch.Tensor()
   self.fccgradInputT = torch.Tensor()
   self.falacc     = torch.Tensor():double()
   self.falmacc    = torch.Tensor():long()
   self.falgacc    = torch.Tensor():double()
   self.fccacc     = torch.Tensor():double()
   self.fccmacc    = torch.Tensor():long()
   self.fccgacc    = torch.Tensor():double()

   self.fccscale = torch.zeros(B):add(1)
   self.falscale = torch.zeros(B):add(1)

   self.lossT = torch.zeros(B)

   self.inputp = ffi.new(string.format("THFloatTensor*[%d]", self.B))
   self.targetp = ffi.new(string.format("THLongTensor*[%d]", self.B))

   self.cdata = {}
   self.cdata.transitions  = self.transitions:cdata()
   self.cdata.gtransitions = self.gtransitions:cdata()
   self.cdata.fccgtransitions = self.fccgtransitions:cdata()
   self.cdata.Tt = self.Tt:cdata()
   self.cdata.TNt = self.TNt:cdata()
   self.cdata.lossT = self.lossT:cdata()
   self.cdata.fccscale = self.fccscale:cdata()
   self.cdata.falscale = self.falscale:cdata()
end

function BatchAutoSegCriterionC:realloc(T, TN)

   self.T  = math.max(T, self.T)
   self.TN = math.max(TN, self.TN)

   print(string.format('[BatchAutoSegCriterion (C based): ' ..
                        'reallocating with B=%d T=%d N=%d TN=%d]',
                        self.B, self.T, self.N, self.TN))

   self.falacc:resize(self.B, self.T, self.TN)
   self.cdata.falacc = self.falacc:cdata()
   self.falmacc:resize(self.B, self.T, self.N)
   self.cdata.falmacc = self.falmacc:cdata()
   self.falgacc:resize(self.B, self.T, self.TN)
   self.cdata.falgacc = self.falgacc:cdata()

   self.fccacc:resize(self.B, self.T, self.N)
   self.cdata.fccacc = self.fccacc:cdata()
   self.fccmacc:resize(self.B, self.T, self.N)
   self.cdata.fccmacc = self.fccmacc:cdata()
   self.fccgacc:resize(self.B, self.T, self.N)
   self.cdata.fccgacc = self.fccgacc:cdata()

   self.gradInputT:resize(self.B, self.T, self.N)
   self.cdata.gradInputT = self.gradInputT:cdata()
   self.fccgradInputT:resize(self.B, self.T, self.N)
   self.cdata.fccgradInputT = self.fccgradInputT:cdata()
end

function BatchAutoSegCriterionC:getB(input, target)
   local B = 0
   for i = 1, #input do
      if input[i]:numel() > 0 then
         B = i
      end
   end
   return B
end

function BatchAutoSegCriterionC:updateOutput(input, target)
   local N = self.N
   local B = self:getB(input, target)
   assert(B <= self.B, 'input larger than batchsize')
   for i = 0, B-1 do
      self.inputp[i]  = input[i+1]:cdata()
      self.targetp[i] = target[i+1]:cdata()
   end
   for i = 1, B do
      self.Tt[i] = input[i]:size(1)
   end
   for i = 1, B do
      self.TNt[i] = target[i]:size(1)
   end
   for i = 1, B do
      self.fccscale[i] = self.scale(input[i], target[i])
   end
   self.falscale:copy(self.fccscale):mul(-1)
   if self.Tt:max() > self.T or self.TNt:max() > self.TN then
      self:realloc(self.Tt:max(), self.TNt:max())
   end
   self.falgacc:zero()
   self.fccgacc:zero()
   self.falacc:zero()
   self.falmacc:zero()
   self.fccacc:zero()
   self.fccmacc:zero()

   C.asgbatchfw(self.inputp,
                self.targetp,
                self.cdata.transitions,
                self.cdata.falacc,
                self.cdata.falmacc,
                self.cdata.falgacc,
                self.cdata.fccacc,
                self.cdata.fccmacc,
                self.cdata.fccgacc,
                self.cdata.falscale,
                self.cdata.fccscale,
                self.cdata.Tt,
                N,
                self.cdata.TNt,
                self.cdata.lossT,
                B)


   local loss = {}
   for i = 1, B do
      loss[i] = self.lossT[i]
   end
   self.output = loss
   return loss
end

function BatchAutoSegCriterionC:viterbi(input)
   error("AutoSegCriterion (C based) does not support viterbi")
end

function BatchAutoSegCriterionC:zeroGradParameters()
   self.gtransitions:zero()
end

function BatchAutoSegCriterionC:updateGradInput(input, target)
   local N = self.N
   local B = self:getB(input, target)
   assert(B <= self.B, 'input larger than batchsize')
   for i = 1, B do
      self.Tt[i] = input[i]:size(1)
   end
   for i = 1, B do
      self.TNt[i] = target[i]:size(1)
   end
   for i = 1, B do
      self.fccscale[i] = self.scale(input[i], target[i])
   end
   self.falscale:copy(self.fccscale):mul(-1)
   for i = 0, B-1 do
      self.inputp[i]  = input[i+1]:cdata()
      self.targetp[i] = target[i+1]:cdata()
   end
   self.fccgtransitions:zero()
   self.gradInputT:zero()
   self.fccgradInputT:zero()
   C.asgbatchbw(self.inputp,
             self.targetp,
             self.cdata.transitions,
             self.cdata.gradInputT,
             self.cdata.fccgradInputT,
             self.cdata.gtransitions,
             self.cdata.fccgtransitions,
             self.cdata.falacc,
             self.cdata.falmacc,
             self.cdata.falgacc,
             self.cdata.fccacc,
             self.cdata.fccmacc,
             self.cdata.fccgacc,
             self.cdata.falscale,
             self.cdata.fccscale,
             self.cdata.Tt,
             N,
             self.cdata.TNt,
             B)
   self.gtransitions:add(self.fccgtransitions)
   self.gradInputT:add(self.fccgradInputT)
   for i = 1, B do
      self.gradInput[i] = self.gradInputT[i]:narrow(1, 1, self.Tt[i])
   end
   return self.gradInput
end

function BatchAutoSegCriterionC:updateParameters(lr)
   for i = 1, self.B do
      self.transitions:add(-lr, self.gtransitions[i])
   end
end

function BatchAutoSegCriterionC:parameters()
   return {self.transitions}, {self.gtransitions}
end

function BatchAutoSegCriterionC:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

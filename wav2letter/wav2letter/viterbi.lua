local utils = require 'wav2letter.utils'
local C = utils.C

local Viterbi = torch.class('nn.Viterbi', 'nn.Criterion')

function Viterbi:realloc(T)
   print(string.format('[Viterbi: reallocating with T=%d N=%d]', T, self.N))

   self.T = T

   self.acc:resize(self.N)
   self.accp:resize(self.N)
   self.macc:resize(self.T, self.N)
end

function Viterbi:__init(N, scale, initsize)
   self.scale = scale or function(input, target) return 1 end
   self.T = 0
   self.N = N
   self.initsize = initsize or 2000
   self.transitions = torch.zeros(N, N)
   self.m      = torch.zeros(N)
   self.path = torch.Tensor():long()

   self.acc    = torch.Tensor()
   self.accp   = torch.Tensor()
   self.macc    = torch.Tensor():long()
end

function Viterbi:viterbi(input)
   local T = input:size(1)
   local N = self.N
   if T > self.T then
      self:realloc(math.max(T, self.initsize))
   end
   self.path:resize(T)
   local score = C.fccviterbi(
      self.acc:cdata(),
      self.macc:cdata(),
      self.accp:cdata(),
      self.path:cdata(),
      input:cdata(),
      self.transitions:cdata(),
      T,
      N
   )
   return self.path, score
end

function Viterbi:parameters()
   return {self.transitions}, {self.gtransitions}
end

function Viterbi:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

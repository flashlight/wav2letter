local DecoderCriterion = torch.class('nn.DecoderCriterion', 'nn.Criterion')
local argcheck = require 'argcheck'

local function logadd(t)
   local m = t:max()
   t = t:clone()
   return m + math.log(t:add(-m):exp():sum())
end

local function dlogadd(t)
   local m = t:max()
   local g = t:clone()
   g:add(-m):exp()
   g:div(g:sum())
   return g
end

DecoderCriterion.__init = argcheck{
   noordered = true,
   {name="self", type="nn.DecoderCriterion"},
   {name="decoder", type="table"},
   {name="dopt", type="table"},
   {name="N", type="number"},
   {name="K", type="number"}, -- number of paths to consider
   {name="scale", type="function", default=function(input, target) return 1 end},
   call =
      function(self, decoder, dopt, N, K, scale)
         assert(K > 0)
         assert(N == #decoder.letters+1)
         self.decoder = decoder
         self.dopt = dopt
         self.K = K
         self.scale = scale
         self.fal = nn.ForceAlignCriterion(N, true)
         self.transitions = self.fal.transitions
         self.gtransitions = torch.Tensor(N, N)
         self.gradInput = torch.Tensor()
      end
}

function DecoderCriterion:setWordTarget(words)
   self.__words = words
end

function DecoderCriterion:updateOutput(input, target)
   local scale = self.scale(input, target)
   self.__scale = scale

   -- fal first because of scale(input, target)
   -- viterbi to get the path as hint
   local falpath = self.fal:viterbi(input, target):add(-1) -- we add -1 to be 0-based, as the decoder
   self.__falpath = falpath.new(falpath:size(1)+2):zero()
   self.__falpath:narrow(1, 2, falpath:size(1)):copy(falpath)

   local falscore = self.fal:updateOutput(input, target)
   local lmscore = self.dopt.lmweight*self.decoder.lm:estimate(self.decoder.usridx2lmidx(self.__words))
   lmscore = lmscore + self.dopt.wordscore*self.__words:size(1)
   falscore = falscore + lmscore

   -- we do not clone path (fast but ugly)
   self.__predictions, self.__lpredictions, self.__scores
      = self.decoder(self.dopt, self.transitions, input, self.K)
   local K = #self.__lpredictions
   local hasfalpath = false
   for k=1,K do
      if self.__lpredictions[k]:equal(self.__falpath) then
         hasfalpath = true
         break
      end
   end
   if not hasfalpath then
      table.insert(self.__lpredictions, self.__falpath)
      table.insert(self.__scores, falscore)
   end
   self.__scores = torch.DoubleTensor(self.__scores)
   self.__scores:mul(scale)

   self.output = logadd(self.__scores) - falscore*scale

   -- print('O', K, self.output, hasfalpath, falscore*scale)
   -- print(self.decoder.tensor2string(self.__words))
   -- print(self:decodedstring())

   return self.output
end

function DecoderCriterion:decodedstring()
   return self.decoder.tensor2string(self.__predictions[1])
end

function DecoderCriterion:zeroGradParameters()
   self.fal:zeroGradParameters()
   self.gtransitions:zero()
end

function DecoderCriterion:updateGradInput(input, target)
   local scale = self.__scale -- from updateOutput
   self.gradInput:resizeAs(input):zero()

   self.fal:updateGradInput(input, target)
   self.gradInput:add(-scale, self.fal.gradInput)
   self.gtransitions:add(-scale, self.fal.gtransitions)

   local g = dlogadd(self.__scores)
   local ginput = self.gradInput
   local gtransitions = self.gtransitions
   for k = 1,#self.__lpredictions do
      local g_k = g[k]
      local target = self.__lpredictions[k]
      local idxm1
      local N = target:size(1)-2 -- beware of start/end nodes
      for t=1,N do
         -- beware of start/end nodes -- beware of 0 indexing
         local idx = target[t+1]+1
         ginput[t][idx] = ginput[t][idx] + scale*g_k
         if idxm1 then
            gtransitions[idx][idxm1] = gtransitions[idx][idxm1] + scale*g_k
         end
         idxm1 = idx
      end
   end

   --   print('G', self.gradInput:norm())

   return self.gradInput
end

function DecoderCriterion:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

function DecoderCriterion:parameters()
   return {self.transitions}, {self.gtransitions}
end

function DecoderCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

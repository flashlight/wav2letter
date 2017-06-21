local DecoderCriterion = torch.class('nn.DecoderCriterion', 'nn.Criterion')
local argcheck = require 'argcheck'

DecoderCriterion.__init = argcheck{
   noordered = true,
   {name="self", type="nn.DecoderCriterion"},
   {name="decoder", type="table"},
   {name="dopt", type="table"},
   {name="N", type="number"},
   {name="scale", type="function", default=function(input, target) return 1 end},
   {name="fbw", type="number", default=0},
   call =
      function(self, decoder, dopt, N, scale, fbw)
         assert(fbw >= 0 and fbw <= 1)
         assert(N == #decoder.letters+1)
         self.decoder = decoder
         self.dopt = dopt
         self.__fallbackweight = fbw
         local function falscale(input, target)
            self.__scale = scale(input, target)
            return -self.__scale
         end
         local function fccscale()
            return self.__scale
         end
         self.fal = nn.ForceAlignCriterion(N, true, falscale)
         self.fcc = nn.FullConnectCriterionC(N, false, fccscale)
         self.fcc.transitions = self.fal.transitions
         self.transitions = self.fal.transitions
         self.gtransitions = torch.Tensor(N, N)
         self.gradInput = torch.Tensor()
      end
}

function DecoderCriterion:setWordTarget(words)
   self.__words = words
end

function DecoderCriterion:updateOutput(input, target)
   -- fal first because of scale(input, target)
   -- viterbi to get the path as hint
   local faloutput = self.fal:updateOutput(input, target)
   local scale = self.__scale

   local fbw = self.__fallbackweight
   -- if fbw > 0 then
   local fccoutput = self.fcc:updateOutput(input)
   self.output = fbw*(faloutput+fccoutput)
   -- else
   --    self.output = 0
   -- end
   -- if fbw < 1 then
   -- we need to add the lm score note: substracting lm score from the
   -- decoder is wrong (decoder finds best path with lm score... so fal
   -- might be higher if lm score is removed there)
   local lmoutput = self.dopt.lmweight*self.decoder.lm:estimate(self.decoder.usridx2lmidx(self.__words))
   lmoutput = lmoutput + self.dopt.wordscore*self.__words:size(1)
   lmoutput = lmoutput*(-scale)
   
   -- we do not clone path (fast but ugly)
   local decoutputs
   self.__predictions, self.__lpredictions, decoutputs
      = self.decoder(self.dopt, self.transitions, input)
   local decoutput = scale*decoutputs
   
   self.output = self.output + (1-fbw)*math.max(0, (faloutput + lmoutput) + decoutput)
   self.__uselm = (faloutput + lmoutput) + decoutput > 0
   -- else
   --    self.__uselm = false
   -- end
   return self.output
end

function DecoderCriterion:decodedstring()
   return self.decoder.tensor2string(self.__predictions)
end

function DecoderCriterion:zeroGradParameters()
   self.fal:zeroGradParameters()
   self.fcc:zeroGradParameters()
   self.gtransitions:zero()
end

function DecoderCriterion:updateGradInput(input, target) 
   local fbw = self.__fallbackweight
   self.gradInput:resizeAs(input)
   self.fal:updateGradInput(input, target)
   self.fcc:updateGradInput(input)
   self.gradInput:copy(self.fal.gradInput)
   self.gradInput:add(fbw, self.fcc.gradInput)

   self.gtransitions:add(self.fal.gtransitions)
   self.gtransitions:add(fbw, self.fcc.gtransitions)
   
   if self.__uselm and fbw < 1 then
      local ginput = self.gradInput
      local scale = self.__scale*(1-fbw)
      local target = self.__lpredictions
      local gtransitions = self.gtransitions
      local idxm1
      local N = target:size(1)-2 -- beware of start/end nodes
      for t=1,N do
         -- beware of start/end nodes -- beware of 0 indexing
         local idx = target[t+1]+1
         ginput[t][idx] = ginput[t][idx] + scale
         if idxm1 then
            gtransitions[idx][idxm1] = gtransitions[idx][idxm1] + scale
         end
         idxm1 = idx
      end
   end
   
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

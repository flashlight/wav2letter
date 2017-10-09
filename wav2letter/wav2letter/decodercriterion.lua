local hasbeamer = pcall(require, 'beamer')
if not hasbeamer then
   return
end

local DecoderCriterion = torch.class('nn.DecoderCriterion', 'nn.Criterion')
local argcheck = require 'argcheck'
local ffi = require 'ffi'
local C = require 'beamer.env'.C

ffi.cdef[[
double BMRDecoder_forward(BMRDecoder *decoder, BMRDecoderOptions *opt, float *transitions, float *emissions, long T, long N);
void BMRDecoder_backward(BMRDecoder *decoder, BMRDecoderOptions *opt, float *gtransitions, float *gemissions, long T, long N, double g);
int BMRDecoder_haspath(BMRDecoder *decoder, long *path, long T);
void BMRDecoder_store_hypothesis(BMRDecoder *decoder, long *nhyp_, float *scores_, long *llabels_, long *labels_);
]]

local function logadd(a, b)
   local m = math.max(a, b)
   return m + math.log(math.exp(a-m)+math.exp(b-m))
end

local function dlogadd(a, b)
   local m = math.max(a, b)
   a = math.exp(a-m)
   b = math.exp(b-m)
   local s = a+b
   return a/s, b/s
end

DecoderCriterion.__init = argcheck{
   noordered = true,
   {name="self", type="nn.DecoderCriterion"},
   {name="decoder", type="table"},
   {name="dopt", type="table"},
   {name="N", type="number"},
   {name="scale", type="function", default=function(input, target) return 1 end},
   call =
      function(self, decoder, dopt, N, scale)
         assert(N == #decoder.letters+1)
         self.decoder = decoder
         self.dopt = dopt
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
   self.__falscore = falscore

   local opt = ffi.new('BMRDecoderOptions', self.dopt)
   local decscore = C.BMRDecoder_forward(self.decoder.decoder, opt, self.transitions:data(), input:data(), input:size(1), input:size(2))
   local output
   self.__decscore = decscore
   self.__haspath = (C.BMRDecoder_haspath(self.decoder.decoder, self.__falpath:data(), input:size(1)) == 1)
   if self.__haspath then
      output = decscore
   else
      output = logadd(decscore, falscore)
   end
   self.output = (output - falscore)*scale

   return self.output
end

function DecoderCriterion:labels(labels_, llabels_)
   llabels_ = llabels_ or torch.LongTensor()
   labels_ = labels_ or torch.LongTensor()
   labels_:resize(self.dopt.beamsize, self.__falpath:size(1))
   llabels_:resize(self.dopt.beamsize, self.__falpath:size(1))
   C.BMRDecoder_store_hypothesis(self.decoder.decoder, nil, nil, llabels_:data(), labels_:data())
   return labels_, llabels_
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

   local ginput = self.gradInput
   local gtransitions = self.gtransitions

   local opt = ffi.new('BMRDecoderOptions', self.dopt)
   if self.__haspath then
      C.BMRDecoder_backward(self.decoder.decoder, opt, gtransitions:data(), ginput:data(), input:size(1), input:size(2), scale)
   else
      local gdecscore, gfalscore = dlogadd(self.__decscore, self.__falscore)
      gdecscore = gdecscore * scale
      gfalscore = gfalscore * scale
      C.BMRDecoder_backward(self.decoder.decoder, opt, gtransitions:data(), ginput:data(), input:size(1), input:size(2), gdecscore)
      local target = self.__falpath
      local idxm1
      local N = target:size(1)-2 -- beware of start/end nodes
      for t=1,N do
         -- beware of start/end nodes -- beware of 0 indexing
         local idx = target[t+1]+1
         ginput[t][idx] = ginput[t][idx] + gfalscore
         if idxm1 then
            gtransitions[idx][idxm1] = gtransitions[idx][idxm1] + gfalscore
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

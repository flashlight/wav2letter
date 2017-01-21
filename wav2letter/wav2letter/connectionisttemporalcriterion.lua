local gtn = require 'gtn'

local ConnectionistTemporalCriterion = torch.class('nn.ConnectionistTemporalCriterion', 'nn.Criterion')
--"Alex Graves - Connectionist Temporal Classification: Labelling
--Unsegmented Sequence Data with Recurrent Neural Networks", ICML 2006
--IMPORTANT: This implementation of CTC assumes no repetitions in the
--target transcription.
--That is "Naan" should be preprocessed to "Nan" or less concretely
--"112" should be shortened to "12".
function ConnectionistTemporalCriterion:realloc(T, S)
   print(string.format("[ConnectionistTemporalCriterion: " ..
                       "reallocating with T=%d S=%d", T, S))
   local Sp = 2*S + 1 --Expanded length

   self.emissions = torch.zeros(T, S + 1)
   local emissions_p = self.emissions:data()
   self.zero = self.zero or torch.zeros(1)

   self.gemissions = self.emissions:clone():zero()
   local gemissions_p = self.gemissions:data()
   self.gzero = self.gzero or torch.zeros(1)

   local zero_p = self.zero:data()
   local gzero_p = self.gzero:data()

   local g = gtn.GTN()

   --The individual target characters are surrounded by blank nodes.
   --Each blank node "guards" a suffix of the target transcription.
   --You can either transistion to that suffix through a blank or via
   --it's first character.
   g:addNode(zero_p, gzero_p)
   for t=1,T do
      --There are S char nodes per frame
      --There are S + 1 blank nodes per frame
      g:addNode(emissions_p + (t-1)*(S + 1) + S,
               gemissions_p + (t-1)*(S + 1) + S)
      for s=1,S do
         g:addNode(emissions_p + (t-1)*(S + 1) + s - 1,
                  gemissions_p + (t-1)*(S + 1) + s - 1)
         g:addNode(emissions_p + (t-1)*(S + 1) + S,
                  gemissions_p + (t-1)*(S + 1) + S)
      end
   end

   --Iterate through each node type
   --First the char nodes, then the blank nodes
   g:addEdge(0, 1, zero_p, gzero_p)
   g:addEdge(0, 2, zero_p, gzero_p)
   for t=1,T-1 do
      --Char nodes
      --One edge to the same char, one edge to the next char
      --(if different) and one edge to the blank before the next char
      for s=1,S do
         local a = (t-1)*Sp + 2*s
         g:addEdge(a, a+Sp, zero_p, gzero_p)
         --There is always a next blank
         g:addEdge(a, a+Sp+1, zero_p, gzero_p)
         --The last character does not have a next one. Repeated
         --emissions are treated as the same character.
         if s < S then g:addEdge(a, a+Sp+2, zero_p, gzero_p) end
      end
      --Blank nodes
      --One edge to the same blank and one edge to the next char
      for s=1,S+1 do
         local a = (t-1)*Sp + 2*(s - 1) + 1
         g:addEdge(a, a + Sp, zero_p, gzero_p)
         --The last blank node does not have a next blank node
         if s <= S then g:addEdge(a, a+Sp+1, zero_p, gzero_p) end
      end
   end

   self.T = T
   self.S = S
   self.g = g
end

function ConnectionistTemporalCriterion:__init(N, scale)
   self.scale = scale or function(input, target) return 1 end
   self.N = N
end

--N here is the number of labels + blank
--So if you're training over the english alphabet, N would be
--26 + 1 = 27. By convention the last probability per input frame is
--the blank probability. The target labels should be over
--{1, ..., N-1}. The blank is not a target character!
--That is, target should be of length less than or equal to input,
--and each entry should contain a positive number that is less than N.
function ConnectionistTemporalCriterion:updateOutput(input, target)
   assert(target:dim() == 1, 'Target tensor is not one dimensional')
   assert(input:dim() == 2, 'Input tensor is not two dimensional')
   local T = input:size(1)
   local N = input:size(2)
   local scale = self.scale(input, target)
   assert(self.N == N, 'input tensor has wrong frame size for criterion')
   local S = target:size(1)
   self.T = self.T or 0
   self.S = self.S or 0
   assert(1 <= target:min() and target:max() < N,
                               'target labels out of range')
   if T > self.T or S > self.S then
      self:realloc(math.max(self.T+1, T+1), math.max(self.S, S))
   end
   self.gradInput = torch.zeros(T, N)
   self.emissions:zero()
   self:mapToEmissions(input, target)
   self.last = T*(self.S*2 + 1) + S*2 + 1
   self.output = -scale * self.g:forwardLogAdd(self.last)
   return self.output
end

function ConnectionistTemporalCriterion:updateGradInput(input, target)
   local scale = self.scale(input, target)
   self.gemissions:zero()
   self.g:backwardLogAdd(-scale, self.last)
   self:mapFromGemissions(input, target)
   return self.gradInput
end

function ConnectionistTemporalCriterion:mapToEmissions(input, target)
   local T = input:size(1)
   local N = input:size(2)
   local S = target:size(1)
   for s=1,S do
      self.emissions[{{1, T}, s}] = input[{{}, target[s]}]
   end
   self.emissions[{{1, T}, self.S+1}] = input[{{}, N}]
end

--Outputs best path with blanks removed
function ConnectionistTemporalCriterion:viterbi(input)
   local _, imax = input:max(2)
   local imax_not = imax:clone():zero()
   local ii = 0
   for i = 1, imax:size(1) do
      if imax[i][1] < self.N then
         ii = ii + 1
         imax_not[ii] = imax[i]
      end
   end
   --If the network output consists only of blanks (self.N)
   --output only a single blank, which won't match with the
   --target transcriptions in any way.
   if ii > 0 then
      imax_not = imax_not:resize(ii)
   else
      imax_not = imax_not:resize(1)
      imax_not[1] = self.N
   end
   return imax_not
end

function ConnectionistTemporalCriterion:mapFromGemissions(input, target)
   local T = input:size(1)
   local N = input:size(2)
   local S = target:size(1)
   for s=1,S do
      self.gradInput[{{}, target[s]}] = self.gradInput[{{}, target[s]}]
                                      + self.gemissions[{{1, T}, s}]
   end
   self.gradInput[{{}, N}] = self.gradInput[{{}, N}]
                           + self.gemissions[{{1, T}, self.S+1}]
end

function ConnectionistTemporalCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

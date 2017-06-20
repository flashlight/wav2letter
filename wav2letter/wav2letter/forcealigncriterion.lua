local gtn = require 'gtn'

-- beware, entering wizard level 10 zone

local ForceAlignCriterion = torch.class('nn.ForceAlignCriterion', 'nn.Criterion')

function ForceAlignCriterion:realloc(T, N)
   print(string.format('[ForceAlignCriterion: reallocating with T=%d N=%d]', T, N))
   assert(N <= T, 'T must be >= N')

   local emissions = torch.Tensor(T, N)
   local subtransitions = torch.Tensor(N, 2) -- n^th row contains [n->n, n-1->n]

   local gemissions = torch.Tensor(T, N)
   local gsubtransitions = torch.Tensor(N, 2)

   local emissions_p = emissions:data()
   local subtransitions_p = subtransitions:data()

   local gemissions_p = gemissions:data()
   local gsubtransitions_p = gsubtransitions:data()

   local g = gtn.GTN()

   -- t in [1,T]
   -- n in [1,N]
   -- return nil if no node
   function self:TN2idx(t, n)
      if t < n or t < 1 or n < 1 or n > N or t > T then
         return
      elseif t <= N then
         return (t-1)*t/2+n-1
      else
         return N*(N+1)/2+(t-N-1)*N+n-1
      end
   end

   function self:Tidx2N(t, idx)
      if idx <= N*(N+1)/2 then
         return idx-(t-1)*t/2+1
      else
         return idx-N*(N+1)/2-(t-N-1)*N+1
      end
   end

   for t=1,T do
      for n=1,N do
         if self:TN2idx(t, n) then
            g:addNode(emissions_p+(t-1)*N+n-1, gemissions_p+(t-1)*N+n-1)
         end
      end
   end

   for t=2,T do
      for n=1,N do
         if self:TN2idx(t,n) then
            if self:TN2idx(t-1,n-1) then
               g:addEdge(self:TN2idx(t-1,n-1), self:TN2idx(t,n), subtransitions_p+(n-1)*2+1, gsubtransitions_p+(n-1)*2+1) -- n-1 -> n
            end
            if self:TN2idx(t-1,n) then
               g:addEdge(self:TN2idx(t-1,n), self:TN2idx(t,n), subtransitions_p+(n-1)*2, gsubtransitions_p+(n-1)*2) -- n -> n
            end
         end
      end
   end

   self.emissions = emissions
   self.subtransitions = subtransitions
   self.gemissions = gemissions
   self.gsubtransitions = gsubtransitions
   self.g = g
   self.T = T
   self.N = N
   self.path = torch.LongTensor(g:nNode())
end

function ForceAlignCriterion:__init(N, ismax, scale)
   self.ismax = ismax
   self.scale = scale or function(input, target) return 1 end
   self.NL = N
   self.N = 0
   self.T = 0
   self.transitions = torch.zeros(N, N)
   self.gtransitions = torch.zeros(N, N)
   self.gradInput = torch.Tensor()
end

function ForceAlignCriterion:updateOutput(input, target)
   local T = input:size(1)
   local N = target:size(1)
   assert(input:size(2) == self.NL)
   local scale = self.scale(input, target)
   assert(N <= T, 'not enough frames to handle all labels')
   if T > self.T or N > self.N then
      self:realloc(math.max(self.T, T,2000), math.max(self.N, N,400)) -- DEBUG!!!!
   end
   self:forwardParams(input, target)
   local loss
   if self.ismax then
      loss = scale*self.g:forwardMax(self:TN2idx(T, N))
   else
      loss = scale*self.g:forwardLogAdd(self:TN2idx(T, N))
   end
   self.output = loss
   return loss
end

-- could be in C
function ForceAlignCriterion:forwardParams(input, target)
   local T = input:size(1)
   local N = target:size(1)
   local emissions = self.emissions:narrow(1, 1, T):narrow(2, 1, N)
   local subtransitions = self.subtransitions:narrow(1, 1, N)
   local transitions = self.transitions
   local idxm1
   for i=1,N do
      local idx = target[i]
      emissions:select(2, i):copy(input:select(2, idx))
      subtransitions[i][1] = transitions[idx][idx]
      if idxm1 then
         subtransitions[i][2] = transitions[idx][idxm1]
      end
      idxm1 = idx
   end
end

function ForceAlignCriterion:viterbi(input, target)
   local T = input:size(1)
   local N = target:size(1)
   assert(N <= T, 'not enough frames to handle all labels')
   if T > self.T or N > self.N then
      self:realloc(math.max(self.T, T,2000), math.max(self.N, N,400)) -- DEBUG!!!!
   end
   self:forwardParams(input, target)
   self.path:resize(self.g:nNode())
   local score, size = self.g:forwardMax(self:TN2idx(T, N), self.path:data())
   return self:pathidx2lbl(self.path, size, target), score
end

function ForceAlignCriterion:zeroGradParameters()
   self.gtransitions:zero()
end

function ForceAlignCriterion:updateGradInput(input, target)
   local T = input:size(1)
   local N = target:size(1)
   local gemissions = self.gemissions:narrow(1, 1, T):narrow(2, 1, N)
   local gsubtransitions = self.gsubtransitions:narrow(1, 1, N)
   local scale = self.scale(input, target)
   gemissions:zero()
   gsubtransitions:zero() -- note: self.gsubtransitions is accumulated into self.gtransitions
   if self.ismax then
      self.g:backwardMax(scale, self:TN2idx(T, N))
   else
      self.g:backwardLogAdd(scale, self:TN2idx(T, N))
   end
   self:backwardParams(input, target)
   return self.gradInput
end

-- could be in C
function ForceAlignCriterion:backwardParams(input, target)
   local T = input:size(1)
   local N = target:size(1)
   local NL = self.NL
   local ginput = self.gradInput:resize(T, NL):zero()
   local gemissions = self.gemissions:narrow(1, 1, T)
   local gsubtransitions = self.gsubtransitions:narrow(1, 1, N)
   local gtransitions = self.gtransitions
   local idxm1
   for i=1,N do
      local idx = target[i]
      ginput:select(2, idx):add(gemissions:select(2, i))
      gtransitions[idx][idx] = gtransitions[idx][idx] + gsubtransitions[i][1]
      if idxm1 then
         gtransitions[idx][idxm1] = gtransitions[idx][idxm1] + gsubtransitions[i][2]
      end
      idxm1 = idx
   end
end

function ForceAlignCriterion:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

function ForceAlignCriterion:pathidx2lbl(path, size, target)
   for t=1,size do
      path[t] = target[self:Tidx2N(t, path[t])]
   end
   path:resize(size)
   return path
end

function ForceAlignCriterion:parameters()
   return {self.transitions}, {self.gtransitions}
end

function ForceAlignCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

local gtn = require 'gtn'

local FullConnectCriterion = torch.class('nn.FullConnectCriterion', 'nn.Criterion')

function FullConnectCriterion:realloc(T)
   local N = self.N
   print(string.format('[FullConnectCriterion: reallocating with T=%d N=%d]', T, N))

   local emissions = torch.zeros(T, N)
   local transitions = self.transitions
   local zero = self.zero or torch.zeros(1)

   local gemissions = torch.zeros(T, N)
   local gtransitions = self.gtransitions
   local gzero = self.gzero or torch.zeros(1)

   local emissions_p = emissions:data()
   local transitions_p = transitions:data()
   local zero_p = zero:data()

   local gemissions_p = gemissions:data()
   local gtransitions_p = gtransitions:data()
   local gzero_p = gzero:data()

   local g = gtn.GTN()

   -- t in [1,T]
   -- n in [1,N+1]
   function self:TN2idx(t, n)
      return (t-1)*(N+1)+n
   end

   for t=0,T do
      if t == 0 then
         g:addNode(zero_p, gzero_p)
      else
         for n=1,N do
            g:addNode(emissions_p+(t-1)*N+n-1, gemissions_p+(t-1)*N+n-1)
         end
         g:addNode(zero_p, gzero_p)
      end
   end

   for t=1,T do
      for n=1,N do
         if t==1 then
            g:addEdge(0, self:TN2idx(t,n), zero_p, gzero_p)
         else
            for nm1=1,N do
               g:addEdge(self:TN2idx(t-1,nm1), self:TN2idx(t,n), transitions_p+((n-1)*N+nm1-1), gtransitions_p+((n-1)*N+nm1-1))
            end
         end
      end
      for n=1,N do
         g:addEdge(self:TN2idx(t,n), self:TN2idx(t,N+1), zero_p, gzero_p)
      end
   end


   self.emissions = emissions
   self.zero = zero
   self.gemissions = gemissions
   self.gzero = gzero
   self.g = g
   self.gradInput = torch.Tensor()
   self.T = T
   self.N = N
   self.path = torch.LongTensor(g:nNode())
end

function FullConnectCriterion:__init(N, ismax, scale)
   self.ismax = ismax
   self.scale = scale or function(input, target) return 1 end
   self.T = 0
   self.N = N
   self.transitions = torch.zeros(N, N)
   self.gtransitions = torch.zeros(N, N)
end

function FullConnectCriterion:updateOutput(input, target)
   local T = input:size(1)
   local N = self.N
   local scale = self.scale(input, target)
   if T > self.T then
      self:realloc(math.max(T, 2000)) -- DEBUG!!!!!
   end
   self.emissions:narrow(1, 1, T):copy(input)
   local loss
   if self.ismax then
      loss = scale*self.g:forwardMax((N+1)*T)
   else
      loss = scale*self.g:forwardLogAdd((N+1)*T)
   end
   if target then
      assert(target:size(1) == T, 'input and target do not match')
      loss = loss + self:forwardTarget(input, target, scale)
   end
   self.output = loss
   return loss
end

function FullConnectCriterion:viterbi(input)
   local T = input:size(1)
   local N = self.N
   if T > self.T then
      self:realloc(T)
   end
   self.emissions:narrow(1, 1, T):copy(input)
   local score, size = self.g:forwardMax((N+1)*T, self.path:data())
   return self:pathidx2lbl(self.path, size), score
end

function FullConnectCriterion:zeroGradParameters()
   self.gtransitions:zero()
end

function FullConnectCriterion:updateGradInput(input, target)
   local T = input:size(1)
   local N = self.N
   local scale = self.scale(input, target)
   local gemissions = self.gemissions:narrow(1, 1, T)
   gemissions:zero()
   self.gzero:zero()
   if target then
      self:backwardTarget(input, target, scale)
   end
   if self.ismax then
      self.g:backwardMax(scale, (N+1)*T)
   else
      self.g:backwardLogAdd(scale, (N+1)*T)
   end
   self.gradInput:view(gemissions, T, N)
   return self.gradInput
end

function FullConnectCriterion:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

-- could be in C
function FullConnectCriterion:forwardTarget(input, target, scale)
   local T = input:size(1)
   local N = self.N
   local loss = 0
   local input_p = input:data()
   local target_p = target:data()
   local transitions_p = self.transitions:data()
   local lblm1
   for t=0,T-1 do
      local lbl = target_p[t]-1
      if lbl < 0 or lbl >= N then
         error('invalid target')
      end
      loss = loss - input_p[t*N+lbl]
      if t > 0 then
         loss = loss - transitions_p[lbl*N+lblm1]
      end
      lblm1 = lbl
   end
   return loss*scale
end

-- could be in C
function FullConnectCriterion:backwardTarget(input, target, scale)
   local T = input:size(1)
   local N = self.N
   local gemissions_p = self.gemissions:data()
   local target_p = target:data()
   local gtransitions_p = self.gtransitions:data()
   local lblm1
   for t=0,T-1 do
      local lbl = target_p[t]-1
      gemissions_p[t*N+lbl] = gemissions_p[t*N+lbl] - scale
      if t > 0 then
         gtransitions_p[lbl*N+lblm1] = gtransitions_p[lbl*N+lblm1] - scale
      end
      lblm1 = lbl
   end
end

-- could be in C
function FullConnectCriterion:pathidx2lbl(path, size)
   local path_p = path:data()
   local N = self.N
   for t=1,size-2 do
      local idx = tonumber(path_p[t])
      local lbl = idx % (N+1)
      path_p[t-1] = lbl
   end
   return path:resize(size-2)
end

function FullConnectCriterion:parameters()
   return {self.transitions}, {self.gtransitions}
end

function FullConnectCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

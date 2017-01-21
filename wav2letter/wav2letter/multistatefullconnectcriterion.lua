local gtn = require 'gtn'

local MultiStateFullConnectCriterion = torch.class('nn.MultiStateFullConnectCriterion', 'nn.Criterion')

function MultiStateFullConnectCriterion:realloc(T)
   local N = self.N
   local S = self.S
   print(string.format('[MultiStateFullConnectCriterion: reallocating with T=%d N=%d S=%d]', T, N, S))

   local emissions = torch.zeros(T, N*S)
   local transitions = self.transitions
   local zero = self.zero or torch.zeros(1)

   local gemissions = torch.zeros(T, N*S)
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
   -- n in [1,N]
   -- s in [1,S+1] note: n=1, s=S+1 only for out nodes
   -- [a tout probleme il y a une solution boeuf]
   local tns2idx = torch.IntTensor(T, N*S+1):fill(-1)
   local idx = 0
   for t=1,T do
      for s=1,math.min(t,S) do
         for n=1,N do
            idx = idx + 1
            tns2idx[t][(s-1)*N+n] = idx
         end
      end
      idx = idx + 1
      tns2idx[t][S*N+1] = idx
   end

   local idx2ns = torch.IntTensor(idx)
   idx = 0
   for t=1,T do
      for s=1,math.min(t,S) do
         for n=1,N do
            idx = idx + 1
            idx2ns[idx] = (s-1)*N+n
         end
      end
      idx = idx + 1
      idx2ns[idx] = -1
   end

   function self:TNS2idx(t, n, s)
      local idx = tns2idx[t][(s-1)*N+n]
      assert(idx > 0) -- big trouble
      return idx
   end

   function self:idx2NS(idx)
      local ns = idx2ns[idx]
      assert(ns > 0) -- big trouble
      return ns
   end

   function self:NS2transidx(n,s,nm1,sm1)
      if nm1 then
         return ((s-1)*N+n-1)*N*S+(sm1-1)*N+nm1
      else
         return (s-1)*N+n
      end
   end

   g:addNode(zero_p, gzero_p)
   for t=1,T do
      for s=1,math.min(t,S) do
         for n=1,N do
            g:addNode(emissions_p+(t-1)*N*S+(s-1)*N+n-1, gemissions_p+(t-1)*N*S+(s-1)*N+n-1)
         end
      end
      g:addNode(zero_p, gzero_p)
   end

   for t=1,T do
      for s=1,math.min(t,S) do
         for n=1,N do
            if t==1 then
               g:addEdge(0, self:TNS2idx(t,n,s), zero_p, gzero_p)
            else
               -- in this case (s==1), can be connected to any previous label (except same, see below), s=S
               -- note that the previous connection exists only if t > S
               if s==1 and t > S then
                  for nm1=1,N do
                     if n ~= nm1 then -- except same label: we deal the == case below
                        g:addEdge(self:TNS2idx(t-1,nm1,S), self:TNS2idx(t,n,s), transitions_p+self:NS2transidx(n,s,nm1,S)-1, gtransitions_p+self:NS2transidx(n,s,nm1,S)-1)
                        assert(transitions_p[self:NS2transidx(n,s,nm1,S)-1] == transitions[self:NS2transidx(n,s)][self:NS2transidx(nm1,S)])
                     end
                  end
               end

               -- can be connected to previous same label, s=same or s=same-1
               for sm1=math.max(1,s-1),s do
                  if t-1 >= sm1 then -- previous s exists only in this condition
                     g:addEdge(self:TNS2idx(t-1,n,sm1), self:TNS2idx(t,n,s), transitions_p+self:NS2transidx(n,s,n,sm1)-1, gtransitions_p+self:NS2transidx(n,s,n,sm1)-1)
                  end
               end
            end
            if s == S then -- beware, cannot exit at each t (quoique? enfin, pas les premiers, en fait)
               g:addEdge(self:TNS2idx(t,n,s), self:TNS2idx(t,1,S+1), zero_p, gzero_p)
            end
         end
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

function MultiStateFullConnectCriterion:__init(N, S, ismax, scale)
   self.ismax = ismax
   self.scale = scale or function(input, target) return 1 end
   self.T = 0
   self.N = N
   self.S = S
   self.transitions = torch.zeros(N*S, N*S)
   self.gtransitions = torch.zeros(N*S, N*S)
end

function MultiStateFullConnectCriterion:updateOutput(input, target)
   local T = input:size(1)
   local N = self.N
   local S = self.S
   local scale = self.scale(input, target)
   if T > self.T then
      self:realloc(T)
   end
   self.emissions:narrow(1, 1, T):copy(input)
   local loss
   if self.ismax then
      loss = scale*self.g:forwardMax(self:TNS2idx(T, 1, S+1))
   else
      loss = scale*self.g:forwardLogAdd(self:TNS2idx(T, 1, S+1))
   end
   if target then
      assert(target:size(1) == T, 'input and target do not match')
      loss = loss + self:forwardTarget(input, target, scale)
   end
   self.output = loss
   return loss
end

function MultiStateFullConnectCriterion:viterbi(input)
   local T = input:size(1)
   local N = self.N
   local S = self.S
   if T > self.T then
      self:realloc(T)
   end
   self.emissions:narrow(1, 1, T):copy(input)
   local score, size = self.g:forwardMax(self:TNS2idx(T, 1, S+1), self.path:data())
   return self:pathidx2lbl(self.path, size), score
end

function MultiStateFullConnectCriterion:zeroGradParameters()
   self.gtransitions:zero()
end

function MultiStateFullConnectCriterion:updateGradInput(input, target)
   local T = input:size(1)
   local N = self.N
   local S = self.S
   local scale = self.scale(input, target)
   local gemissions = self.gemissions:narrow(1, 1, T)
   gemissions:zero()
   self.gzero:zero()
   if target then
      self:backwardTarget(input, target, scale)
   end
   if self.ismax then
      self.g:backwardMax(scale, self:TNS2idx(T, 1, S+1))
   else
      self.g:backwardLogAdd(scale, self:TNS2idx(T, 1, S+1))
   end
   self.gradInput:view(gemissions, T, N*S)
   return self.gradInput
end

function MultiStateFullConnectCriterion:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

-- could be in C
function MultiStateFullConnectCriterion:forwardTarget(input, target, scale)
   local T = input:size(1)
   local N = self.N
   local S = self.S
   local loss = 0
   local input_p = input:data()
   local target_p = target:data()
   local transitions_p = self.transitions:data()
   local lblm1
   for t=0,T-1 do
      local lbl = target_p[t]-1
      assert(lbl >= 0 and lbl < N*S)
      loss = loss - input_p[t*N*S+lbl]
      if t > 0 then
         loss = loss - transitions_p[lbl*N*S+lblm1]
      end
      lblm1 = lbl
   end
   return loss*scale
end

-- could be in C
function MultiStateFullConnectCriterion:backwardTarget(input, target, scale)
   local T = input:size(1)
   local N = self.N
   local S = self.S
   local gemissions_p = self.gemissions:data()
   local target_p = target:data()
   local gtransitions_p = self.gtransitions:data()
   local lblm1
   for t=0,T-1 do
      local lbl = target_p[t]-1
      gemissions_p[t*N*S+lbl] = gemissions_p[t*N*S+lbl] - scale
      if t > 0 then
         gtransitions_p[lbl*N*S+lblm1] = gtransitions_p[lbl*N*S+lblm1] - scale
      end
      lblm1 = lbl
   end
end

-- could be in C
function MultiStateFullConnectCriterion:pathidx2lbl(path, size)
   local path_p = path:data()
   for t=1,size-2 do
      local idx = tonumber(path_p[t])
      path_p[t-1] = self:idx2NS(idx)
   end
   return path:resize(size-2)
end

function MultiStateFullConnectCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

local gtn = require 'gtn'

local FullConnectGarbageCriterion = torch.class('nn.FullConnectGarbageCriterion', 'nn.Criterion')

function FullConnectGarbageCriterion:realloc(T)
   local N = self.N
   print(string.format('[FullConnectGarbageCriterion: reallocating with T=%d N=%d]', T, N))

   local emissions = torch.zeros(T, N+1)
   local transitions = self.transitions
   local zero = self.zero or torch.zeros(1)

   local gemissions = torch.zeros(T, N+1)
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
   function self:TN2idx(t, n, islabel)
      if islabel then
         assert(n <= N+1) -- escape
         return (t-1)*(2*N+1)+2*(n-1)+1
      else
         assert(n <= N)
         return (t-1)*(2*N+1)+2*(n-1)+2
      end
   end

   for t=0,T do
      if t == 0 then
         g:addNode(zero_p, gzero_p) -- start
      else
         for n=1,N do
            g:addNode(emissions_p+(t-1)*(N+1)+n-1, gemissions_p+(t-1)*(N+1)+n-1)
            g:addNode(emissions_p+(t-1)*(N+1)+N, gemissions_p+(t-1)*(N+1)+N) -- last output for garbage
         end
         g:addNode(zero_p, gzero_p) -- escape
      end
   end

   for t=1,T do
      for n=1,N do
         if t==1 then
            -- start -> label
            g:addEdge(0, self:TN2idx(t,n,true), zero_p, gzero_p)
            -- print("DEBUG MODE, PLEASE FIX")
            -- -- start -> garbage [debug]
            -- g:addEdge(0, self:TN2idx(t,n,false), zero_p, gzero_p)
         else
            -- label -> label
            local nm1 = n
            g:addEdge(self:TN2idx(t-1,nm1,true), self:TN2idx(t,n,true), transitions_p+((n-1)*N+nm1-1), gtransitions_p+((n-1)*N+nm1-1))

            -- garbage -> label
            for nm1=1,N do
               g:addEdge(self:TN2idx(t-1,nm1,false), self:TN2idx(t,n,true), transitions_p+((n-1)*N+nm1-1), gtransitions_p+((n-1)*N+nm1-1))
            end

            -- label -> garbage
            local nm1 = n
            g:addEdge(self:TN2idx(t-1,nm1,true), self:TN2idx(t,n,false), transitions_p+((n-1)*N+nm1-1), gtransitions_p+((n-1)*N+nm1-1))

            -- garbage -> garbage
            local nm1 = n
            g:addEdge(self:TN2idx(t-1,nm1,false), self:TN2idx(t,n,false), transitions_p+((n-1)*N+nm1-1), gtransitions_p+((n-1)*N+nm1-1))
         end
      end
      for n=1,N do
         -- label -> escape
         g:addEdge(self:TN2idx(t,n,true), self:TN2idx(t,N+1,true), zero_p, gzero_p)
         -- garbage -> escape
         g:addEdge(self:TN2idx(t,n,false), self:TN2idx(t,N+1,true), zero_p, gzero_p)
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

function FullConnectGarbageCriterion:__init(N, ismax, scale)
   self.ismax = ismax
   self.scale = scale or function(input) return 1 end
   self.T = 0
   self.N = N
   self.transitions = torch.zeros(N, N)
   self.gtransitions = torch.zeros(N, N)
end

function FullConnectGarbageCriterion:updateOutput(input)
   local T = input:size(1)
   local N = self.N
   local scale = self.scale(input)
   if T > self.T then
      self:realloc(T)
--      self:realloc(math.max(T, 500)) -- DEBUG!!!!!
   end
   self.emissions:narrow(1, 1, T):copy(input)
   local loss
   if self.ismax then
      loss = scale*self.g:forwardMax(self:TN2idx(T,N+1,true))
   else
      loss = scale*self.g:forwardLogAdd(self:TN2idx(T,N+1,true))
   end
   self.output = loss
   return loss
end

function FullConnectGarbageCriterion:viterbi(input)
   local T = input:size(1)
   local N = self.N
   if T > self.T then
      self:realloc(T)
   end
   self.emissions:narrow(1, 1, T):copy(input)
   local score, size = self.g:forwardMax(self:TN2idx(T,N+1,true), self.path:data())
   self.path:resize(size)
   return self:pathidx2lbl(self.path, size), score
end

function FullConnectGarbageCriterion:zeroGradParameters()
   self.gtransitions:zero()
end

function FullConnectGarbageCriterion:updateGradInput(input)
   local T = input:size(1)
   local N = self.N
   local scale = self.scale(input)
   local gemissions = self.gemissions:narrow(1, 1, T)
   gemissions:zero()
   self.gzero:zero()
   if self.ismax then
      self.g:backwardMax(scale, self:TN2idx(T,N+1,true))
   else
      self.g:backwardLogAdd(scale, self:TN2idx(T,N+1,true))
   end
   self.gradInput:view(gemissions, T, N+1)
   return self.gradInput
end

function FullConnectGarbageCriterion:updateParameters(lr)
   self.transitions:add(-lr, self.gtransitions)
end

-- could be in C
function FullConnectGarbageCriterion:pathidx2lbl(path, size)
   local path_p = path:data()
   local N = self.N
   local n = 0
   assert(path[1] == 0)
   assert(path[size] % (2*N+1) == 0)
   for t=2,size-1 do
      local idx = tonumber(path_p[t-1])
      assert(idx > 0)
      local lbl = (idx-1) % (2*N+1) + 1
      lbl = math.ceil(lbl/2)
      assert(lbl <= N)
      path_p[n] = lbl
      n = n + 1
   end
   return path:resize(n)
end

function FullConnectGarbageCriterion:parameters()
   return {self.transitions}, {self.gtransitions}
end

function FullConnectGarbageCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

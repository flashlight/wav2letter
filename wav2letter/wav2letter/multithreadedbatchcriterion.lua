local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

local MultiThreadedBatchCriterion = torch.class('nn.MultiThreadedBatchCriterion', 'nn.Criterion')

--Only tensors can be shared
function MultiThreadedBatchCriterion:__init(batchsize, shares, class, ...)
   local args = {...}
   self.batchsize = batchsize
   self.shares = shares
   local upvalues = {}
   for i, v in pairs(shares) do
      upvalues[v] = torch.Tensor()
   end
   self.pool = threads.Threads(
      batchsize,
      function(threadid)
         require 'wav2letter'
         torch.setdefaulttensortype('torch.FloatTensor')
         crit = nn[class](unpack(args))
         for k, v in pairs(upvalues) do
            if crit[k] then
               if v:numel() == 0 then
                  v:resizeAs(crit[k]):copy(crit[k])
               end
               crit[k]:set(v)
            else
               crit[k] = v
            end
         end
      end
   )
   for k, v in pairs(upvalues) do
      if v:numel() > 0 then
         self[k] = v
      end
   end
   self.pool:synchronize()
   self.pool:specific(true)
   self.args = args
   self.class = class
end

function MultiThreadedBatchCriterion:updateOutput(input, target)
   self.output = {}
   for i = 1, #input do
      if input[i]:numel() > 0 then
         self.pool:addjob(
            i,
            function()
               --print('mtbc ' .. i .. ' crit.transitions:sum() ' .. crit.transitions:sum())
               return crit:updateOutput(input[i], target[i])
            end,
            function(out)
               self.output[i] = out
            end
         )
      end
   end
   self.pool:synchronize()
   return self.output
end

function MultiThreadedBatchCriterion:viterbi(input)
   self.paths = {}
   self.scores = {}
   for i = 1, #input do
      if input[i]:numel() > 0 then
         self.pool:addjob(
            i,
            function()
               return crit:viterbi(input[i])
            end,
            function(path, score)
               self.paths[i] = path
               self.scores[i] = score
            end
         )
      end
   end
   self.pool:synchronize()
   return self.paths, self.scores
end

function MultiThreadedBatchCriterion:zeroGradParameters()
   for i = 1, self.batchsize do
      self.pool:addjob(
         i,
         function()
            if crit.zeroGradParameters then
               crit:zeroGradParameters()
            end
         end
      )
   end
   self.pool:synchronize()
end

function MultiThreadedBatchCriterion:updateGradInput(input, target)
   self.gradInput = {}
   for i = 1, #input do
      if input[i]:numel() > 0 then
         self.pool:addjob(
            i,
            function()
               return crit:updateGradInput(input[i], target[i])
            end,
            function(out)
               self.gradInput[i] = out
            end
         )
      end
   end
   self.pool:synchronize()
   return self.gradInput
end

function MultiThreadedBatchCriterion:updateParameters(lr)
   for i = 1, self.batchsize do
      self.pool:addjob(
         i,
         function()
            if crit.updateParameters then
               crit:updateParameters(lr)
            end
         end
      )
      self.pool:synchronize() --Careful! Write to transitions atomically
   end
end

function MultiThreadedBatchCriterion:clone()
   return nn.MultiThreadedBatchCriterion(self.batchsize, self.shares, self.class, unpack(self.args))
end

function MultiThreadedBatchCriterion:share(layer, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      for i = 1, self.batchsize do
         local t = layer[v]
         assert(t, 'given variable ' .. v .. ' cannot be shared')
         self.pool:addjob(
            i,
            function()
               if crit[v] ~= nil then
                  crit[v]:set(t)
               end
            end
         )
      end
      self.pool:synchronize()
      if self[v] ~= nil then
         self[v]:set(layer[v])
      end
   end
   return self
end

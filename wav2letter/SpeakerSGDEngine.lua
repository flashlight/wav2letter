local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'

require 'nn'

local SpeakerSGDEngine, SGDEngine = torch.class('tnt.SpeakerSGDEngine', 'tnt.SGDEngine', tnt)

SpeakerSGDEngine.train = argcheck{
   {name="self", type="tnt.SpeakerSGDEngine"},
   {name="network", type="nn.Module"},
   {name="criterion", type="nn.Criterion"},
   {name="iterator", type="tnt.DatasetIterator"},
   {name="lr", type="number"},
   {name="lrcriterion", type="number", defaulta="lr"},
   {name="maxepoch", type="number", default=1000},
   call =
      function(self, network, criterion, iterator, lr, lrcriterion, maxepoch)
         local state = {
            network = network,
            criterion = criterion,
            iterator = iterator,
            lr = lr,
            lrcriterion = lrcriterion,
            maxepoch = maxepoch,
            sample = {},
            epoch = 0, -- epoch done so far
            t = 0, -- samples seen so far
            training = true
         }

         self.hooks("onStart", state)
         while state.epoch < state.maxepoch do
            state.network:training()

            self.hooks("onStartEpoch", state)
            for sample in state.iterator() do
               state.sample = sample
               self.hooks("onSample", state)

               state.network:forward(sample.input)
               self.hooks("onForward", state)
               state.criterion:forward(state.network.output, sample.target)
               self.hooks("onForwardCriterion", state)

               state.network:zeroGradParameters()
               if state.criterion.zeroGradParameters then
                  state.criterion:zeroGradParameters()
               end

               state.criterion:backward(state.network.output, sample.target)
               self.hooks("onBackwardCriterion", state)
               state.network:backward(sample.input, state.criterion.gradInput)
               self.hooks("onBackward", state)

              --  assert(state.lrcriterion >= 0, 'lrcriterion should be positive or zero')
              --  if state.lrcriterion > 0 and state.criterion.updateParameters then
              --     state.criterion:updateParameters(state.lrcriterion)
              --  end
              --  assert(state.lr >= 0, 'lr should be positive or zero')
              --  if state.lr > 0 then
              --     local ct_idx = 0
              --     for i=1, #state.network.modules do
              --       if torch.type(state.network.modules[i]) == 'nn.ConcatTable' then
              --         ct_idx = i
              --       end
              --     end
              --     -- state.network.modules[ct_idx].modules[2]:updateParameters(state.lr)
              --  end            
               state.t = state.t + 1
               self.hooks("onUpdate", state)
            end
            state.epoch = state.epoch + 1
            self.hooks("onEndEpoch", state)
         end
         self.hooks("onEnd", state)
      end
}

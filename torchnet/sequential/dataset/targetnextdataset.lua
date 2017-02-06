-- Copyright 2004-present Facebook. All Rights Reserved.
--[[
--
-- This dataset produces augments samples with successive samples, assuming that
-- the underlying dataset represents a sequence. Samples at the end of the
-- underlying dataset will be discarded if they don't have a corresponding
-- target.
--
-- This is useful for generating targets for language modeling tasks.
--
--]]

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local TargetNextDataset, Dataset =
    torch.class('tnt.TargetNextDataset', 'tnt.Dataset', tnt)


TargetNextDataset.__init = argcheck{
    {name='self', type='tnt.TargetNextDataset'},
    {name='dataset', type='tnt.Dataset'},
    {name='input', type='string', default = 'input'},
    {name='target', type='string', default = 'target'},
    {name='step', type='number', default = 1},
    call =
        function(self, dataset, input, target, step)
            Dataset.__init(self)
            self._dataset = dataset
            self._input = input
            self._target = target
            self._step = step
        end
}

TargetNextDataset.size = argcheck{
    {name='self', type='tnt.TargetNextDataset'},
    call = function(self)
        return math.max(self._dataset:size() - self._step, 0)
    end
}

TargetNextDataset.get = argcheck{
    {name='self', type='tnt.TargetNextDataset'},
    {name='idx', type='number'},
    call = function(self, idx)
       assert(idx >= 1 and idx <= self:size(), 'index out of bound')
       assert(idx == math.floor(idx), 'index must be an integer')
       local sample = self._dataset:get(idx)
       assert(not sample[self._target], 'target field is already set')
       local nextt = self._dataset:get(idx + self._step)

       assert(nextt[self._input] ~= nil,
            string.format('no such key: %s', self._input))
       sample[self._target] = nextt[self._input]
       return sample
    end
}

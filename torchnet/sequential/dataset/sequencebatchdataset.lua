-- Copyright 2004-present Facebook. All Rights Reserved.
--[[
--
-- This dataset produces multiple, parallel sequences from a single
-- sequence represented by the underlying dataset.
--
--]]

local tnt = require 'torchnet'
local argcheck = require 'argcheck'

local SequenceBatchDataset, Dataset =
    torch.class('tnt.SequenceBatchDataset', 'tnt.Dataset', tnt)


SequenceBatchDataset.__init = argcheck{
    {name='self', type='tnt.SequenceBatchDataset'},
    {name='dataset', type='tnt.Dataset'},
    {name='batchsize', type='number'},
    {name='pad', type='number', default=1},
    {name='type', type='string', default='torch.LongTensor'},
    {name='policy', type='string', default='pad-remainder'},
    call =
        function(self, dataset, batchsize, pad, type, policy)
            Dataset.__init(self)
            self._dataset = dataset
            self._batchsize = batchsize
            self._pad = pad
            self._type = type
            self._policy = policy
            assert(self._batchsize >= 1, 'batchsize out of bound')
            self:size()  -- check policy
        end
}

SequenceBatchDataset.size = argcheck{
    {name='self', type='tnt.SequenceBatchDataset'},
    call = function(self)
        if self._policy == 'pad-remainder' then
            return math.ceil(self._dataset:size() / self._batchsize)
        elseif self._policy == 'skip-remainder' then
            return math.floor(self._dataset:size() / self._batchsize)
        else
            error('invalid policy (pad-remainder | skip-remainder expected)')
        end
    end
}

SequenceBatchDataset.get = argcheck{
    {name='self', type='tnt.SequenceBatchDataset'},
    {name='idx', type='number'},
    call = function(self, idx)
        assert(idx >= 1 and idx <= self:size(), 'index out of bound')
        assert(idx == math.floor(idx), 'index must be an integer')

        local samples = torch.Tensor():type(self._type):resize(self._batchsize)
        local step = self:size()
        local maxidx = self._dataset:size()
        for i = 1, self._batchsize do
            local j = ((i-1) * step) + idx
            if j > maxidx then
                samples[i] = self._pad
            else
                samples[i] = self._dataset:get(j)
            end
        end
        return samples
    end
}

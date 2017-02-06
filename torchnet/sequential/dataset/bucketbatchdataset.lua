-- Copyright 2004-present Facebook. All Rights Reserved.
--[[
--
-- This class resamples a dataset using BucketSortedDataset and yields batches
-- from within buckets. Similar to tnt.BatchDataset 'policy' is used to control
-- the behavior regarding remaining samples in a bucket that do not fill up a
-- whole batch.
--
--]]

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local transform = require 'torchnet.transform'

local BucketBatchDataset, _ =
    torch.class('tnt.BucketBatchDataset', 'tnt.Dataset', tnt)

BucketBatchDataset.__init = argcheck{
    {name='self', type='tnt.BucketBatchDataset'},
    {name='dataset', type='tnt.Dataset'},
    {name='batchsize', type='number'},
    -- The policy is applied per-bucket
    {name='policy', type='string', default='include-last'},
    {name='merge', type='function', opt=true},
    {name='samplesize', type='function',
        default=function(dataset, idx) return #dataset:get(idx) end},
    {name='resolution', type='number', default=1},
    {name='shuffle', type='boolean', default=false},
    call = function(self, dataset, batchsize, policy, merge, samplesize,
        resolution, shuffle)
        assert(batchsize > 0 and math.floor(batchsize) == batchsize,
            'batchsize should be a positive integer number')
        assert(resolution > 0, 'resolution should be positive')
        self.__bucketds = tnt.BucketSortedDataset{
            dataset = dataset,
            resolution = resolution,
            samplesize = samplesize,
            shuffle = shuffle,
        }
        self.__makebatch = transform.makebatch{merge=merge}

        -- NOTE: It's assumed that BucketSortedDataset does not change its
        -- buckets after construction.
        self.__batches = {}
        for _, bucket in ipairs(self.__bucketds:bucketRanges()) do
            local offset = bucket.offset
            local nextBucketOffset = bucket.size + bucket.offset
            while offset + batchsize <= nextBucketOffset do
                table.insert(self.__batches,
                    {bstart=offset, bend=offset + batchsize - 1}
                )
                offset = offset + batchsize
            end
            if policy == 'include-last' and offset ~= nextBucketOffset then
                table.insert(self.__batches,
                    {bstart=offset, bend=nextBucketOffset - 1}
                )
            end
        end
     end
}

BucketBatchDataset.size = argcheck{
    {name='self', type='tnt.BucketBatchDataset'},
    call = function(self)
        return #self.__batches
    end
}

BucketBatchDataset.get = argcheck{
    {name='self', type='tnt.BucketBatchDataset'},
    {name='idx', type='number'},
    call = function(self, idx)
        assert(idx >= 1 and idx <= self:size(), 'index out of bound')
        local samples = {}
        local batch = self.__batches[idx]
        for i = batch.bstart, batch.bend do
            table.insert(samples, self.__bucketds:get(i))
        end
        samples = self.__makebatch(samples)
        collectgarbage()
        return samples
    end
}

BucketBatchDataset.resampleInBuckets = argcheck{
    {name='self', type='tnt.BucketBatchDataset'},
    call = function(self)
        self.__bucketds:resampleInBuckets()
    end
}

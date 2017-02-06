-- Copyright 2004-present Facebook. All Rights Reserved.
--[[
--
-- This is a resampling dataset that clusters samples into buckets based on
-- their size. By customizing the `samplesize` function and setting the
-- `resolution` argument, various bucketing and sharding patterns can be
-- implemented. TODO: example
--
-- Optionally, the dataset can permute the order of examples within individual
-- buckets. This way, successive samples will be of similar size while some
-- level of randomnes is achieved. Note that the the order of buckets is
-- constant so that this dataset can be easily wired up with tnt.BatchDataset
-- and tnt.ShuffleDataset to produce random mini-batches.
--
--]]

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local tensorvector = require 'tensorvector'

local BucketSortedDataset, ResampleDataset =
    torch.class('tnt.BucketSortedDataset', 'tnt.ResampleDataset', tnt)


BucketSortedDataset.__init = argcheck{
    {name='self', type='tnt.BucketSortedDataset'},
    {name='dataset', type='tnt.Dataset'},
    {name='samplesize', type='function',
        default=function(dataset, idx) return #dataset:get(idx) end},
    {name='resolution', type='number', default=1},
    {name='shuffle', type='boolean', default=false},
    call = function(self, dataset, samplesize, resolution, shuffle)
        -- Initialize buckets
        local buckets = {}
        local bucketidx = {}
        local size = dataset:size()
        for i = 1, size do
            local bucket = math.floor(samplesize(dataset, i) / resolution)
            if not buckets[bucket] then
                buckets[bucket] = tensorvector.new_long()
                table.insert(bucketidx, bucket)
            end
            buckets[bucket][#buckets[bucket] + 1] = i
        end
        table.sort(bucketidx)

        self.__index = torch.LongTensor(size)
        self.__buckets = {}  -- Stores {offset, size} to self.__index
        local n = 1
        for _, bucketIndex in ipairs(bucketidx) do
            local bucket = buckets[bucketIndex]
            self.__index:narrow(1, n, #bucket):copy(bucket:getTensor())
            table.insert(self.__buckets, {offset=n, size=#bucket})
            n = n + #bucket
        end

        buckets, bucketidx = nil, nil
        collectgarbage()

        self.__perm = self.__index:clone()
        ResampleDataset.__init(self, {
            dataset = dataset,
            sampler = function(dataset, idx)
                return self.__perm[idx]
            end,
            size = size
        })

        if shuffle then
            self:resampleInBuckets()
        end
    end
}

-- This should rather be called resample(), but this class is used in pipelines
-- together with tnt.SortedDataset. In this case, this function would be
-- shadowed by the respective function in tnt.SortedDataset, rendering it
-- unreachable from exec() on the top-level dataset/iterator.
BucketSortedDataset.resampleInBuckets = argcheck{
    {name='self', type='tnt.BucketSortedDataset'},
    call = function(self)
        for _, bucket in ipairs(self.__buckets) do
            local offset, size = bucket.offset, bucket.size
            local bucketPerm = torch.randperm(size):long() + offset - 1
            self.__perm:narrow(1, offset, size):copy(
                self.__index:index(1, bucketPerm)
            )
        end
    end
}

BucketSortedDataset.bucketRanges = argcheck{
    {name='self', type='tnt.BucketSortedDataset'},
    call = function(self)
        return self.__buckets
    end
}

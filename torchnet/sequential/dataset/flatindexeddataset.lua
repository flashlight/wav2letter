-- Copyright 2004-present Facebook. All Rights Reserved.
--[[
--
-- This dataset provides a "flat" view of the storage that would usually back a
-- tnt.IndexedDataset. In other words, this dataset disregards the index on disk
-- and provides access to the individual data elements.
--
-- For some use cases, it's still handy to know which element belongs to which
-- index entry (which could be a sequence ID, for example). If `indices` is set
-- to true, get(i) will return both the element value and the position of the
-- entry in the dataset index.
--
--]]

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local FlatIndexedDataset, _ =
    torch.class('tnt.FlatIndexedDataset', 'tnt.Dataset', tnt)


FlatIndexedDataset.__init = argcheck{
    {name='self', type='tnt.FlatIndexedDataset'},
    {name='indexfilename', type='string'},
    {name='datafilename', type='string'},
    {name='mmap', type='boolean', default=false},
    {name='mmapidx', type='boolean', default=false},
    {name='indices', type='boolean', default=false},
    call = function(self, indexfilename, datafilename, mmap, mmapidx, indices)
        self._reader = tnt.IndexedDatasetReader{
            indexfilename = indexfilename,
            datafilename = datafilename,
            mmap = mmap,
            mmapidx = mmapidx,
        }
        self._view = self._reader.data:narrow(1, 1, self:size())
        self._indices = indices
    end
}

FlatIndexedDataset.size = argcheck{
    {name='self', type='tnt.FlatIndexedDataset'},
    call = function(self)
        return self._reader.datoffsets[self._reader.N + 1]
    end
}

FlatIndexedDataset.get = argcheck{
    {name='self', type='tnt.FlatIndexedDataset'},
    {name='idx', type='number'},
    call = function(self, idx)
        assert(idx >= 1 and idx <= self:size(), 'index out of bound')
        if self._indices then
            local seqidx = FlatIndexedDataset._lowerBound(
                self._reader.datoffsets, idx - 1
            )
            return self._view[idx], seqidx
        else
            return self._view[idx]
        end
    end
}

FlatIndexedDataset._lowerBound = argcheck{
    doc = [[
Performs a search in T, a sorted 1D tensor, and returns an index in T
so that T[index] <= a < T[index+1].

If a is smaller than T[1] or larger than T[#T], behavior is undefined.
Similarly, if T contains non-unique elements, there's no guarantee wrt
which of those will be found.
]],
    {name='T', type='torch.*Tensor'},
    {name='a', type='number'},
    call = function(T, a)
        local istart, iend, idx = 1, T:nElement(), 0
        while istart < iend do
            idx = math.floor((istart + iend)/2)
            if T[idx + 1] <= a then
                istart = idx + 1
            elseif T[idx] > a then
                iend = idx - 1
            else
                break
            end
        end
        if istart == iend then
            idx = istart
        end
        return idx
    end
}

-- Copyright 2004-present Facebook. All Rights Reserved.
--[[
--
-- This iterator truncates samples produced by its underlying dataset or
-- iterator if they exceed the specified maximum size. In this case, multiple
-- smaller samples are produced. If the samples of the underlying dataset are
-- tables, samples that have been split will have an additional '_split' flag
-- set. Two additional flags make post-processing more convenient: `_hasnext` is
-- set on all parts of a split except for the last; `_cont` is set on all parts
-- except for the first one.
--
-- This iterator can be used to enforce a limit on backpropagation through
-- time.
--
--]]

local tnt = require 'torchnet'
local argcheck = require 'argcheck'

local TruncatedDatasetIterator, DatasetIterator =
    torch.class('tnt.TruncatedDatasetIterator', 'tnt.DatasetIterator', tnt)


TruncatedDatasetIterator.__init = argcheck{
    {name='self', type='tnt.TruncatedDatasetIterator'},
    {name='dataset', type='tnt.Dataset'},
    {name='maxsize', type='number', opt=true},
    {name='minsize', type='number', opt=true},
    {name='dimension', type='number', default=1},
    {name='fields', type='table', default={}},
    call = function(self, dataset, maxsize, minsize, dimension, fields)
        DatasetIterator.__init(self, dataset)
        self:_setup(maxsize, minsize, nil, dimension, fields)
    end
}

TruncatedDatasetIterator.__init = argcheck{
    {name='self', type='tnt.TruncatedDatasetIterator'},
    {name='iterator', type='tnt.DatasetIterator'},
    {name='maxsize', type='number', opt=true},
    {name='minsize', type='number', opt=true},
    {name='dimension', type='number', default=1},
    {name='fields', type='table', default={}},
    overload =  TruncatedDatasetIterator.__init,
    call = function(self, iterator, maxsize, minsize, dimension, fields)
        DatasetIterator.__init(self, iterator)
        self:_setup(maxsize, minsize, nil, dimension, fields)
    end
}

TruncatedDatasetIterator.__init = argcheck{
    {name='self', type='tnt.TruncatedDatasetIterator'},
    {name='dataset', type='tnt.Dataset'},
    {name='maxsizefn', type='function'},
    {name='dimension', type='number', default=1},
    {name='fields', type='table', default={}},
    overload =  TruncatedDatasetIterator.__init,
    call = function(self, dataset, maxsizefn, dimension, fields)
        DatasetIterator.__init(self, dataset)
        self:_setup(nil, nil, maxsizefn, dimension, fields)
    end
}

TruncatedDatasetIterator.__init = argcheck{
    {name='self', type='tnt.TruncatedDatasetIterator'},
    {name='iterator', type='tnt.DatasetIterator'},
    {name='maxsizefn', type='function'},
    {name='dimension', type='number', default=1},
    {name='fields', type='table', default={}},
    overload =  TruncatedDatasetIterator.__init,
    call = function(self, iterator, maxsizefn, dimension, fields)
        DatasetIterator.__init(self, iterator)
        self:_setup(nil, nil, maxsizefn, dimension, fields)
    end
}

TruncatedDatasetIterator._setup = argcheck{
    {name='self', type='tnt.TruncatedDatasetIterator'},
    {name='maxsize', type='number', opt=true},
    {name='minsize', type='number', opt=true},
    {name='maxsizefn', type='function', opt=true},
    {name='dimension', type='number'},
    {name='fields', type='table'},
    call = function(self, maxsize, minsize, maxsizefn, dimension, fields)
        self.maxsize = maxsize or math.huge
        self.minsize = minsize or 0
        self.maxsizefn = maxsizefn
        self.dimension = dimension

        self.field_map = {}
        for _,v in pairs(fields) do
            self.field_map[v] = true
        end

        -- Base class iterator function
        self.base_run = self.run

        -- Iterator state. self.sample is the current sample obtained from the
        -- underlying dataset/iterator.
        self.sample = nil
        self.sample_size = -1
        self.sample_part_offset = -1
        self.yield_parts = false
        self.run = self:_run()
    end
}

-- Slice tensor alpng dim
local function partTensor(T, dim, offset, maxsize)
    local index = math.min(offset, T:size(dim) + 1)
    local size = math.min(maxsize, T:size(dim) - index + 1)
    return T:narrow(dim, index, size)
end

local function partTable(tbl, dim, offset, maxsize, fields)
    local part = {}
    for k,v in pairs(tbl) do
        local typename = torch.typename(v)
        if fields[k] and typename and typename:match('Tensor') then
            part[k] = partTensor(v, dim, offset, maxsize)
        else
            part[k] = v
        end
    end
    return part
end

-- Returns sample size along dim and asserts that split samples are of equal
-- length.
local function sampleSize(sample, dim, fields)
    local sz = nil
    if type(sample) == 'table' then
        for k,v in pairs(sample) do
            local typename = torch.typename(v)
            if fields[k] and typename and typename:match('Tensor')
                and v:nDimension() >= dim then
                if not sz then
                    sz = v:size(dim)
                else
                    assert(sz == v:size(dim),
                        'sample values must be of equal length')
                end
            end
        end
    else
        sz = sample:size(dim)
    end
    return sz
end

function TruncatedDatasetIterator:_run()
    return function()
        local next_from_base = self.base_run()

        local part = function()
            -- Assemble current part
            local part = nil
            if type(self.sample) == 'table' then
                part = partTable(self.sample, self.dimension,
                    self.sample_part_offset, self.maxsize, self.field_map)

                -- Mark this sample as a continuation of a previous one
                part._cont = self.sample_part_offset > 1
                part._hasnext = self.sample_part_offset + self.maxsize
                    <= self.sample_size
                part._split = true
            else
                part = partTensor(self.sample, self.dimension,
                    self.sample_part_offset, self.maxsize)
            end

            -- Advance to next part
            self.sample_part_offset = self.sample_part_offset + self.maxsize
            if self.sample_part_offset > self.sample_size then
                self.yield_parts = false
            end
            return part
        end

        local getnext = function()
            if self.yield_parts then
                return part()
            end

            self.sample = next_from_base()
            if not self.sample then
                return self.sample
            end
            if self.maxsizefn then
                self.maxsize = self.maxsizefn(self.sample)
            end
            if self.maxsize == math.huge then
                return self.sample
            end

            -- Check if sample exceeds the maximum size
            self.sample_size = sampleSize(self.sample, self.dimension,
                self.field_map) or self.minsize
            if self.sample_size <= self.maxsize then
                return self.sample
            end

            self.yield_parts = true
            self.sample_part_offset = 1
            return part()
        end

        return function()
            local sample, cursize
            -- Get samples until minsize criteria is met
            repeat
                sample = getnext()
                if not sample then
                    break
                end
                cursize = sampleSize(sample, self.dimension, self.field_map)
                    or self.minsize
            until cursize >= self.minsize
            return sample
        end
    end
end

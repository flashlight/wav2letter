require 'fb.luaunit'
local tnt = require 'torchnet.sequential'

local ds = tnt.TableDataset{data = {
    torch.rand(5),
    torch.rand(3),
    torch.rand(1),
    torch.rand(7),
}}

function testSimple()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsize = 3,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s:size(1))
    end
    assertEquals(#partlens, its)
end

function testSimpleFn()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsizefn = function(s) return 3 end,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s:size(1))
    end
    assertEquals(#partlens, its)
end

function testIterator()
    local it = tnt.TruncatedDatasetIterator{
        iterator = tnt.DatasetIterator{
            dataset = ds,
        },
        maxsize = 3,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s:size(1))
    end
    assertEquals(#partlens, its)
end

function testIterator()
    local it = tnt.TruncatedDatasetIterator{
        iterator = tnt.DatasetIterator{
            dataset = ds,
        },
        maxsizefn = function(s)
            if s:size(1) == 7 then
                return math.huge  -- don't split
            end
            return 3
        end,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 7}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s:size(1))
    end
    assertEquals(#partlens, its)
end

function testSingle()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsize = 1,
    }

    local its = 0
    for s in it() do
        its = its + 1
        assertEquals(1, s:size(1))
    end
    assertEquals(16, its)
end

function testTableContSplit()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample) return {x = sample} end
        },
        maxsize = 3,
        fields = {'x'},
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    local conts = {false, true, false, false, false, true, true}
    local nexts = {true, false, false, false, true, true, false}
    local splits = {true, true, false, false, true, true, true}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s.x:size(1))
        assertEquals(conts[its], s._cont == true)
        assertEquals(nexts[its], s._hasnext == true)
        assertEquals(splits[its], s._split == true)
    end
    assertEquals(#partlens, its)
end

function testTableNoFields()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample) return {x = sample} end
        },
        maxsize = 3,
    }

    local its = 0
    for s in it() do
        its = its + 1
        assertEquals(ds:get(its):totable(), s.x:totable())
    end
    assertEquals(ds:size(), its)
end

function testExclude()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample) return {x = sample, y = sample} end
        },
        maxsize = 3,
        fields = {'x'}
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    local excllens = {5, 5, 3, 1, 7, 7, 7}
    local cons = {false, true, false, false, false, true, true}
    local nexts = {true, false, false, false, true, true, false}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s.x:size(1))
        assertEquals(excllens[its], s.y:size(1))
        assertEquals(cons[its], s._cont == true)
        assertEquals(nexts[its], s._hasnext == true)
    end
    assertEquals(#partlens, its)
end

function testMixedLength()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        dataset = tnt.TransformDataset{
            dataset = ds,
            transform = function(sample)
                return {x = sample, y = torch.cat(sample, sample)}
            end
        },
        maxsize = 3,
        fields = {'x', 'y'}
    }

    -- This will trigger an assertion since x and y are of different
    -- length
    assertError(it())
end

function testBatch()
    local it = tnt.TruncatedDatasetIterator{
        dataset = tnt.TableDataset{data = {
            torch.rand(1, 5),
            torch.rand(2, 3),
            torch.rand(3, 1),
            torch.rand(4, 7),
        }},
        maxsize = 3,
        dimension = 2,
    }

    local its = 0
    local partlens = {3, 2, 3, 1, 3, 3, 1}
    local batchlens = {1, 1, 2, 3, 4, 4, 4}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s:size(2))
        assertEquals(batchlens[its], s:size(1))
    end
    assertEquals(#partlens, its)
end

function testMinSize()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
        maxsize = 3,
        minsize = 2,
    }

    local its = 0
    local partlens = {3, 2, 3, 3, 3}
    for s in it() do
        its = its + 1
        assertEquals(partlens[its], s:size(1))
    end
    assertEquals(#partlens, its)
end

function testIdentity()
    local it = tnt.TruncatedDatasetIterator{
        dataset = ds,
    }

    local its = 0
    for s in it() do
        its = its + 1
        assertEquals(ds:get(its):totable(), s:totable())
    end
    assertEquals(ds:size(), its)
end

LuaUnit:main()

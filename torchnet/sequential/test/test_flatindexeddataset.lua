require 'fb.luaunit'
local tnt = require 'torchnet.sequential'
local paths = require 'paths'

local data = {
    torch.IntTensor({1, 2, 3, 4, 5}),
    torch.IntTensor({6, 7}),
    torch.IntTensor({8, 9, 10, 11}),
    torch.IntTensor({12}),
    torch.IntTensor({13, 14, 15, 16, 17, 18}),
    torch.IntTensor({19}),
    torch.IntTensor({20}),
    torch.IntTensor({21, 22}),
    torch.IntTensor({23, 24, 25, 26}),
}

function testEqual()
    -- XXX A temporary directory function would be great
    local dest = os.tmpname()
    local writer = tnt.IndexedDatasetWriter{
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
        type = 'int',
    }

    for _, t in ipairs(data) do
        writer:add(t)
    end
    writer:close()

    local field = paths.basename(dest)
    local ds = tnt.IndexedDataset{
        fields = {field},
        path = paths.dirname(dest),
    }
    assertEquals(#data, ds:size())

    local fds = tnt.FlatIndexedDataset{
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
        indices = true,
    }
    local totals = 0
    for i = 1, ds:size() do
        totals = totals + ds:get(i)[field]:nElement()
    end
    assertEquals(totals, fds:size())

    -- Check elements and indices to original data
    local j = 1
    for i = 1, ds:size() do
        local s = ds:get(i)[field]:view(-1)
        for k = 1, s:nElement() do
            local v, idx = fds:get(j)
            assertEquals(s[k], v)
            assertEquals(i, idx)
            j = j + 1
        end
    end
end

function testLowerBound()
    local t = torch.IntTensor({1, 2, 5, 7, 8})
    local lbound = tnt.FlatIndexedDataset._lowerBound

    assertEquals(1, lbound(t, 1))
    assertEquals(2, lbound(t, 2))
    assertEquals(2, lbound(t, 3))
    assertEquals(2, lbound(t, 4))
    assertEquals(3, lbound(t, 5))
    assertEquals(3, lbound(t, 6))
    assertEquals(4, lbound(t, 7))
    assertEquals(5, lbound(t, 8))
    -- Out-of-bounds queries are not handled by this function.

    assertEquals(0, lbound(torch.IntTensor(), 1))
end

LuaUnit:main()

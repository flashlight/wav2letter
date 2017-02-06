require 'fb.luaunit'
local tnt = require 'torchnet.sequential'

function testSimple()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}
    local bsz = 4
    local ds = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = bsz,
    }

    assertEquals(math.ceil(src:size() / bsz), ds:size())
    assertEquals(bsz, ds:get(1):size(1))
    assertEquals(bsz, ds:get(1):nElement())
    assertEquals({1, 5, 9, 13}, ds:get(1):totable())
    assertEquals({2, 6, 10, 14}, ds:get(2):totable())
    assertEquals({3, 7, 11, 1}, ds:get(3):totable())
    assertEquals({4, 8, 12, 1}, ds:get(4):totable())
    assertError(tnt.SequenceBatchDataset.get, ds, 5)
    assertError(tnt.SequenceBatchDataset.get, ds, 0)
end

function testTypes()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}
    local bsz = 4

    -- Try a few common types
    local types = {'torch.IntTensor', 'torch.LongTensor', 'torch.FloatTensor',
        'torch.DoubleTensor'}
    for _,t in ipairs(types) do
        assertEquals(t, tnt.SequenceBatchDataset{
                dataset = src,
                batchsize = bsz,
                type = t,
            }:get(1):type()
        )
    end

    local dsF = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = bsz,
        type = 'nosuchtype',
    }
    -- Access should fail when trying to create a tensor with the nonexistent
    -- type
    assertError(tnt.SequenceBatchDataset.get, dsF, 1)
end

function testExtremes()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}

    assertError(tnt.SequenceBatchDataset.new, src, 0)

    local dsN = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size(),
    }
    assertEquals(1, dsN:size())
    assertEquals(src:size(), dsN:get(1):nElement())

    local ds2N = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() * 2,
    }
    assertEquals(1, ds2N:size())
    assertEquals(src:size() * 2, ds2N:get(1):nElement())

    local ds2NS = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() * 2,
        policy = 'skip-remainder',
    }
    assertEquals(0, ds2NS:size())
end

function testPolicy()
    local src = tnt.TableDataset{data = torch.linspace(1, 14, 14):totable()}

    local dsSL = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() - 1,
        policy = 'skip-remainder',
    }
    assertEquals(1, dsSL:size())
    assertEquals(torch.linspace(1, 13, 13):totable(), dsSL:get(1):totable())

    local dsIL = tnt.SequenceBatchDataset{
        dataset = src,
        batchsize = src:size() - 1,
        policy = 'pad-remainder',
        pad = 0,
    }
    assertEquals(2, dsIL:size())
    assertEquals(
        torch.cat(torch.range(1, 13, 2), torch.zeros(6)):totable(),
        dsIL:get(1):totable()
    )
    assertEquals(
        torch.cat(torch.range(2, 14, 2), torch.zeros(6)):totable(),
        dsIL:get(2):totable()
    )
end

LuaUnit:main()

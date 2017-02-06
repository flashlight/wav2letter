require 'fb.luaunit'
local tnt = require 'torchnet.sequential'

function testSimple()
    local data = {
        "a", "b", "c",
        "aa", "bb", "cc",
        "aaaa", "bbbb", "cccc",
        "foobar",
    }

    torch.manualSeed(1234)
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 2,
        merge = function(tbl) return {s=table.concat(tbl.s, ',')} end,
        samplesize = function(ds, i) return #ds:get(i).s end,
        shuffle = true
    }
    assertEquals(7, ds:size())
    assertEquals("a,c", ds:get(1).s)
    assertEquals("b", ds:get(2).s)
    assertEquals("aa,cc", ds:get(3).s)
    assertEquals("bb", ds:get(4).s)
    assertEquals("cccc,bbbb", ds:get(5).s)
    assertEquals("aaaa", ds:get(6).s)
    assertEquals("foobar", ds:get(7).s)

    ds:resampleInBuckets()
    assertEquals("a,c", ds:get(1).s)
    assertEquals(nil, ds:get(2).s:find(','))
    assertEquals(nil, ds:get(4).s:find(','))
    assertEquals(nil, ds:get(6).s:find(','))
    assertEquals(nil, ds:get(7).s:find(','))

    local ds2 = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 4,
        merge = function(tbl) return {s=table.concat(tbl.s, ',')} end,
        resolution = 3,
        samplesize = function(ds, i) return #ds:get(i).s end,
    }
    assertEquals(4, ds2:size())
    assertEquals("a,b,c,aa", ds2:get(1).s)
    assertEquals("bb,cc", ds2:get(2).s)
    assertEquals("aaaa,bbbb,cccc", ds2:get(3).s)
    assertEquals("foobar", ds2:get(4).s)
end

function testIncludeLast()
    local data = torch.linspace(0, 99, 100):long()
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data:totable()},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 20,
        resolution = 50,
        merge = function(tbl) return {s=torch.LongTensor(tbl.s)} end,
        samplesize = function(ds, i) return ds:get(i).s end,
        policy = 'include-last',
    }

    assertEquals(6, ds:size())
    assertEquals(20, ds:get(1).s:size(1))
    assertEquals(20, ds:get(2).s:size(1))
    assertEquals(10, ds:get(3).s:size(1))
    assertEquals(20, ds:get(4).s:size(1))
    assertEquals(20, ds:get(5).s:size(1))
    assertEquals(10, ds:get(6).s:size(1))
end

function testSkipLast()
    local data = torch.linspace(0, 99, 100):long()
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data:totable()},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 20,
        resolution = 50,
        merge = function(tbl) return {s=torch.LongTensor(tbl.s)} end,
        samplesize = function(ds, i) return ds:get(i).s end,
        policy = 'skip-last',
    }

    assertEquals(4, ds:size())
    assertEquals(20, ds:get(1).s:size(1))
    assertEquals(20, ds:get(2).s:size(1))
    assertEquals(20, ds:get(3).s:size(1))
    assertEquals(20, ds:get(4).s:size(1))
end

local function doTestEven(policy)
    local data = torch.linspace(0, 11, 12):long()
    local ds = tnt.BucketBatchDataset{
        dataset = tnt.TransformDataset{
            dataset = tnt.TableDataset{data = data:totable()},
            transform = function(sample) return {s=sample} end,
        },
        batchsize = 2,
        resolution = 4,
        merge = function(tbl) return {s=torch.LongTensor(tbl.s)} end,
        samplesize = function(ds, i) return ds:get(i).s end,
        policy = policy,
    }

    assertEquals(6, ds:size())
    assertEquals(2, ds:get(1).s:size(1))
    assertEquals(2, ds:get(2).s:size(1))
    assertEquals(2, ds:get(3).s:size(1))
    assertEquals(2, ds:get(4).s:size(1))
    assertEquals(2, ds:get(5).s:size(1))
    assertEquals(2, ds:get(6).s:size(1))
end

function testEvenIncludeLast()
    doTestEven('include-last')
end

function testEvenSkipLast()
    doTestEven('skip-last')
end

LuaUnit:main()

require 'fb.luaunit'
local tnt = require 'torchnet.sequential'

function testSimple()
    local data = {
        "a", "b", "c",
        "foobar",
        "aa", "bb",
        "aaaa", "bbbb", "cccc",
        "cc",
    }

    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data},
    }
    assertEquals(10, ds:size())
    assertEquals("a", ds:get(1))
    assertEquals("b", ds:get(2))
    assertEquals("c", ds:get(3))
    assertEquals("aa", ds:get(4))
    assertEquals("bb", ds:get(5))
    assertEquals("cc", ds:get(6))
    assertEquals("aaaa", ds:get(7))
    assertEquals("bbbb", ds:get(8))
    assertEquals("cccc", ds:get(9))
    assertEquals("foobar", ds:get(10))

    torch.manualSeed(1234)
    local ds1 = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data},
        shuffle = true,
    }
    assertEquals(10, ds1:size())
    assertEquals("a", ds1:get(1))
    assertEquals("c", ds1:get(2))
    assertEquals("b", ds1:get(3))
    assertEquals("aa", ds1:get(4))
    assertEquals("cc", ds1:get(5))
    assertEquals("bb", ds1:get(6))
    assertEquals("cccc", ds1:get(7))
    assertEquals("bbbb", ds1:get(8))
    assertEquals("aaaa", ds1:get(9))
    assertEquals("foobar", ds1:get(10))

    ds1:resampleInBuckets()
    assertEquals("a", ds1:get(1))
    assertEquals("foobar", ds1:get(10))

    local ds2 = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data},
        resolution = 3,
        shuffle = true,
    }
    assertEquals(10, ds2:size())
    assertEquals("bb", ds2:get(1))
    assertEquals("aa", ds2:get(2))
    assertEquals("cc", ds2:get(3))
    assertEquals("b", ds2:get(4))
    assertEquals("a", ds2:get(5))
    assertEquals("c", ds2:get(6))
    assertEquals("aaaa", ds2:get(7))
    assertEquals("bbbb", ds2:get(8))
    assertEquals("cccc", ds2:get(9))
    assertEquals("foobar", ds2:get(10))
end

local function getAllSamples(ds)
    local samples = {}
    for sample in tnt.DatasetIterator(ds)() do
        table.insert(samples, sample)
    end
    return torch.LongTensor(samples)
end

function testSingleBucket()
    local data = torch.linspace(1, 100, 100):long()
    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data:totable()},
        samplesize = function(ds, i) return 1 end,
        shuffle = true,
    }

    torch.manualSeed(3)
    ds:resampleInBuckets()
    assertEquals(2, data:eq(getAllSamples(ds)):sum())

    torch.manualSeed(4)
    ds:resampleInBuckets()
    assertEquals(0, data:eq(getAllSamples(ds)):sum())
end

function testOneElemBuckets()
    local data = torch.linspace(1, 100, 100):long()
    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data:totable()},
        samplesize = function(ds,i) return ds:get(i) end,
        shuffle = true,
    }

    assertEquals(100, data:eq(getAllSamples(ds)):sum())
    ds:resampleInBuckets()
    assertEquals(100, data:eq(getAllSamples(ds)):sum())
end

function testNegativeLength()
    local data = torch.linspace(1, 100, 100):long()
    local ds = tnt.BucketSortedDataset{
        dataset = tnt.TableDataset{data = data:totable()},
        samplesize = function(ds, i) return -ds:get(i) end,
        shuffle = true,
    }

    local datar = torch.sort(data, 1, true)
    assertEquals(100, datar:eq(getAllSamples(ds)):sum())
end

LuaUnit:main()

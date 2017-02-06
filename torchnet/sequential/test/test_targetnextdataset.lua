require 'fb.luaunit'
local tnt = require 'torchnet.sequential'

function testSimple()
    local src = tnt.TransformDataset{
        dataset = tnt.TableDataset{data = torch.linspace(1, 10, 10):totable()},
        transform = function(sample) return {input = sample} end
    }
    local step = 2
    local ds = tnt.TargetNextDataset{
        dataset = src,
        step = step,
    }

    assertEquals(src:size() - step, ds:size())
    assertEquals({input = 1, target = 3}, ds:get(1))
    assertEquals({input = 2, target = 4}, ds:get(2))
    assertEquals({input = 3, target = 5}, ds:get(3))
    assertError(tnt.TargetNextDataset.get, ds, 0)
    assertError(tnt.TargetNextDataset.get, ds, src:size())
end

function testExtremes()
    local src = tnt.TransformDataset{
        dataset = tnt.TableDataset{data = torch.linspace(1, 10, 10):totable()},
        transform = function(sample) return {input = sample} end
    }

    local ds0 = tnt.TargetNextDataset{
        dataset = src,
        step = 0
    }
    assertEquals(src:size(), ds0:size())
    assertEquals({input = 1, target = 1}, ds0:get(1))

    local dsN = tnt.TargetNextDataset{
        dataset = src,
        step = src:size()-1
    }
    assertEquals(1, dsN:size())
    assertEquals({input = 1, target = 10}, dsN:get(1))

    local ds2N = tnt.TargetNextDataset{
        dataset = src,
        step = src:size() * 2
    }
    assertEquals(0, ds2N:size())
end

LuaUnit:main()

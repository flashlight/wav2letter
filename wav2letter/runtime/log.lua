-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local argcheck = require "argcheck"
local log = {}

log.status = argcheck{
   noordered = true,
   {name="meters", type="table"},
   {name="state", type="table", default={epoch=0, lr=0, lrcriterion=0}},
   {name="verbose", type="boolean", default=false},
   {name="separator", type="string", default=" "},
   {name="date", type="boolean", default=false},
   {name="opt", type="table"},
   {name="reduce", type="function", default=function(x) return x end},
   call =
      function(meters, state, verbose, separator, date, opt, reduce)
         local header = {}
         local status = {}
         local ERR = opt.target:sub(1, 1):upper() .. "ER"
         local function item(key, format, value)
            table.insert(header, key)
            if verbose then
               table.insert(status, string.format("%s " .. format, key, value))
            else
               table.insert(status, string.format(format, value))
            end
         end
         if date then
            item("date", "%s", os.date("%Y-%m-%d"))
            item("time", "%s", os.date("%H:%M:%S"))
         end
         item("epoch", "%10.2f", state.epoch)
         item("lr", "%4.6f", state.lr)
         item("lrcriterion", "%4.6f", state.lrcriterion)
         item("runtime", os.date("!%X", meters.runtime:value()))
         item("ms(bch)", "%4d", meters.timer:value()*1000)
         item("ms(smp)", "%4d", meters.sampletimer:value()*1000)
         item("ms(net)", "%4d", meters.networktimer:value()*1000)
         item("ms(crt)", "%4d", meters.criteriontimer:value()*1000)
         item("loss", "%10.5f", reduce(meters.loss:value()))
         if opt.seg then
            item("train ferr", "%5.2f", reduce(meters.trainframeerr:value()))
         end
         item(string.format("train-%s", ERR), "%5.2f", meters.trainedit:reduce(reduce):value())
         for name, meter in pairs(meters.validedit) do
            item(string.format("%s-%s", name, ERR), "%5.2f", meter:reduce(reduce):value())
         end
         for name, meter in pairs(meters.testedit) do
            item(string.format("%s-%s", name, ERR), "%5.2f", meter:reduce(reduce):value())
         end
         if opt.wer then
            item("train-WER", "%5.2f", meters.wordedit:reduce(reduce):value())
            for name, meter in pairs(meters.validwordedit) do
               item(string.format("%s-WER", name), "%5.2f", meter:reduce(reduce):value())
            end
            for name, meter in pairs(meters.testwordedit) do
               item(string.format("%s-WER", name), "%5.2f", meter:reduce(reduce):value())
            end
         end
         if opt.bmrwer then
            for name, meter in pairs(meters.validbmrwordedit) do
               item(string.format("%s-bWER", name), "%5.2f", meter:reduce(reduce):value())
            end
            for name, meter in pairs(meters.testbmrwordedit) do
               item(string.format("%s-bWER", name), "%5.2f", meter:reduce(reduce):value())
            end
         end
         local stats = meters.stats:value()
         item("aisz", "%03d", reduce(stats["isz"]/stats["n"]))
         item("atsz", "%03d", reduce(stats["tsz"]/stats["n"]))
         item("mtsz", "%03d", reduce(stats["maxtsz"]))
         item("h", "%7.2f", reduce(stats["isz"]/opt.samplerate/3600))
         return table.concat(status, separator), table.concat(header, separator)
      end
}

log.print2file = argcheck{
   {name="file", type="torch.File"},
   {name="date", type="boolean", default=false},
   {name="stdout", type="boolean", default=false},
   call =
      function(file, date, stdout)
         local oprint = print -- get the current print function lazily
         function print(...)
            local n = select("#", ...)
            local arg = {...}
            if stdout then
               oprint(os.date("%Y-%m-%d %H:%M:%S"), ...)
            end
            local str = {}
            if date then
               table.insert(str, os.date("%Y-%m-%d %H:%M:%S"))
            end
            for i=1,n do
               table.insert(str, tostring(arg[i]))
            end
            table.insert(str, "\n")
            file:seekEnd()
            file:writeString(table.concat(str, " "))
            file:synchronize()
         end
      end
}

return log

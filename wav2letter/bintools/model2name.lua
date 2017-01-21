#!/home/cpuhrsch/torch/install/bin/luajit
require 'torch'
local function readName(path)
   local f = torch.DiskFile(path, 'r', true)
   if f ~= nil then
      f = f:binary()
      local dbg = f:readObject()
      if dbg then
         print(dbg.name)
      end
      f:close()
   end
end

for line in io.stdin:lines() do
   readName(line)
end

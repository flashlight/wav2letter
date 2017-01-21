#!/home/cpuhrsch/torch/install/bin/luajit
require 'torch'
local function readName(path)
   local f = torch.DiskFile(path, 'r', true)
   if f ~= nil then
      f = f:binary()
      f:readObject()
      local opt = f:readObject()
      if opt then
         local cmd = "fry flow-gpu"
         local cpu_core = math.max(opt.batchsize*5 or 5, opt.nthread or 2)
         local resources = string.format("'{\"gpu\": %d, \"cpu_core\": %d, \"ram_gb\": %d}'", opt.batchsize or 1, cpu_core, opt.batchsize*40 or 40)
         local environment = string.format("'{\"OMP_NUM_THREADS\":\"%d\"}'", math.max(cpu_core - 1))
         cmd = cmd .. " --resources " .. resources
         cmd = cmd .. " --environment " .. environment
         cmd = cmd .. " --binary-type local /home/cpuhrsch/fbcode/buck-out/gen/deeplearning/projects/wav2letter/train.lex "
         cmd = cmd .. " -continue " .. path
         print(cmd)
      end
      f:close()
   end
end

for line in io.stdin:lines() do
   readName(line)
end

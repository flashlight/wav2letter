local tnt = require 'torchnet'
require 'wav2letter'
local readers = require 'wav2letter.readers'
require 'xlua'

local tokens =
   function(fname)
      local f = io.open(fname)
      local txt = f:read('*all')
      f:close()
      local t = {}
      for _ in txt:gmatch('(%S+)') do
         table.insert(t, 0)
      end
      return torch.Tensor(t)
   end
local featuredict = {
   {name = "flac", reader = readers.audio{}},
   {name = "wrd",  reader = tokens},
   {name = "phn",  reader = tokens},
   {name = "ltr",  reader = tokens},
}

for _, path in ipairs(arg) do
   local features = {}
   print("| valid features:")
   for _, feature in ipairs(featuredict) do
      local fname = paths.concat(path, string.format("%09d.%s", 1, feature.name))
      local f = io.open(fname)
      if f then
         f:close()
         table.insert(features, feature)
         print("   *", feature.name)
      end
   end
   local dataset = tnt.NumberedFilesDataset{
      path = path,
      features = features
   }
   local N = dataset:size()
   for n=1,N do
      local sample = dataset:get(n)
      for feature, data in pairs(sample) do
         local fname = paths.concat(path, string.format("%09d.%s", n, feature .. "sz"))
         local f = io.open(fname, "w")
         f:write(data:nDimension() > 0 and data:size(1) or 0)
         f:close()
      end
      xlua.progress(n, N)
   end
end

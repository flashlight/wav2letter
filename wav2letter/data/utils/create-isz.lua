local tnt = require 'torchnet'
require 'wav2letter'
local readers = require 'wav2letter.readers'
require 'xlua'

local feature = "flac"

for _, path in ipairs(arg) do
   local dataset = tnt.NumberedFilesDataset{
      path = path,
      features = {
         {
            name = feature,
            alias = "wav",
            reader = readers.audio{},
         }
      }
   }
   local N = dataset:size()
   for n=1,N do
      local wav = dataset:get(n).wav
      local fname = paths.concat(path, string.format("%09d.%s", n, "isz"))
      local f = io.open(fname, "w")
      f:write(wav:size(1))
      f:close()
      xlua.progress(n, N)
--      print(i, fname, wav:size(1))
   end
end

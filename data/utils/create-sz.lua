-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

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
   for _, feature in ipairs(featuredict) do
      print(string.format("[checking feature <%s>]", feature.name))
      local status, dataset = pcall(tnt.NumberedFilesDataset, {
         path = path,
         features = {feature}})
      if status then
         local N = dataset:size()
         for n=1,N do
            local sample = dataset:get(n)
            for feature, data in pairs(sample) do
               local fname = dataset:filename(n, feature .. "sz")
               local f = io.open(fname, "w")
               f:write(data:nDimension() > 0 and data:size(1) or 0)
               f:close()
            end
            if (n-1) % 100 == 0 or n == N then
               xlua.progress(n, N)
            end
         end
      end
   end
end

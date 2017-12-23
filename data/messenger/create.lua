-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local paths = require 'paths'
local xlua = require 'xlua'
local torch = require 'torch'
local tnt = require 'torchnet'

--Blobs
local Index = require('fbcode.blobs.lua.index')
local IndexLoader = require('fbcode.blobs.lua.index-loader')
local PathResolver = require('fbcode.blobs.lua.path-resolver')
local StreamMultiReader = require('fbcode.blobs.lua.stream-multi-reader')

local function makeReader(mtbl)
   local index = Index:new(
       IndexLoader:newInMemoryOrdered():addSubloader(
           IndexLoader:newFile(mtbl)
       )
   )
   return StreamMultiReader:newFromIndex{
       index = index,
       num_threads = 1,
       path_resolver = PathResolver:new():addSearchPath(
           paths.dirname(mtbl)),
   }
end

local function createProgress(N)
   local n = 0
   return function ()
      n = n + 1
      xlua.progress(n, N)
   end
end

local function createidx(dat, mtbl, dst)
   print('Creating reader')
   local reader = makeReader(mtbl)

   print('Counting datapoints')
   local N = 0
   for line in io.lines(dat) do
      N = N + 1
   end

   print(string.format('Sifting through %d datapoints', N))
   local progress = createProgress(N)

   local inputidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/input.idx", dst),
      datafilename = string.format("%s/input.bin", dst),
      type = "byte"
   }

   local targetidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/target.idx", dst),
      datafilename = string.format("%s/target.bin", dst),
      type = "byte"
   }

   local filenameidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/filename.idx", dst),
      datafilename = string.format("%s/filename.bin", dst),
      type = "byte"
   }

   print(string.format('| writing %s...', dst))
   local size = 0
   for line in io.lines(dat) do
      local lbl = line:match("{TEXT ([^}]+)}")
      local adc  = line:match("{ADC ([^}]+)}")
      local keys = {adc}
      local data = reader:findKeysAndRead{keys = keys}
      if data[keys[1]] then
         inputidx:add(torch.ByteTensor(torch.ByteStorage():string(data[keys[1]].value)))

         lbl = lbl:gsub('^%s+', '')
         lbl = lbl:gsub('%s+$', '')
         lbl = lbl:lower()
         targetidx:add(torch.ByteTensor(torch.ByteStorage():string(lbl)))

         filenameidx:add(torch.ByteTensor(torch.ByteStorage():string(adc)))

         progress()
         size = size + 1
      end
   end

   print(string.format("%d/%d useable examples", size, N))

   inputidx:close()
   targetidx:close()
   filenameidx:close()
end

assert(#arg == 3, string.format('usage: %s <dat file> <mtbl file> <dst dir>', arg[0]))
local dat = arg[1]
local mtbl = arg[2]
local dst = arg[3]

print('Creating index')
createidx(dat, mtbl, dst)

-- create letters list
print('| creating letter list...')
local alltargets = tnt.IndexedDataset{
   path = dst,
   fields = {"target"}
}

local letters = {}
for i=1,alltargets:size() do
   local sample = alltargets:get(i)
   sample.target:apply(
      function(letter)
         letter = string.char(letter)
         if not letters[letter] then
            table.insert(letters, letter)
            letters[letter] = #letters
         end
      end
   )
end
table.sort(
   letters,
   function(a, b)
      return a < b
   end
)
local f = io.open(string.format("%s/letters.lst", dst), "w")
for _, letter in ipairs(letters) do
   f:write(letter .. '\n')
end
f:close()

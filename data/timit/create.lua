-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- timit phones
-- core test set setup

local paths = require 'paths'
local sndfile = require 'sndfile'

require 'lfs'

local function copytoflac(src, dst, check)
   local f = sndfile.SndFile(src)
   local i = f:info()
   assert(i.subformat == 'PCM16')
   local d = f:readShort(i.frames)
   f:close()
   local f = sndfile.SndFile(
      dst,
      "w",
      {
         samplerate=i.samplerate,
         format='FLAC',
         subformat='PCM16',
         channels=1
      }
   )
   f:writeShort(d)
   f:close()
   if check then
      local g = sndfile.SndFile(dst)
      local gd = g:readShort(g:info().frames)
      g:close()
      gd:add(-1, d)
      assert(gd:min() == 0 and gd:max() == 0)
   end
end

local function fileapply(path, regex, closure)
   for filename in lfs.dir(path) do
      if filename ~= '.' and filename ~= '..' then
         filename = path .. '/' .. filename
         if lfs.attributes(filename, 'mode') == 'directory' then
            fileapply(filename, regex, closure)
         else
            if filename:match(regex) then
               closure(filename)
            end
         end
      end
   end
end


local function createdataset(src, dst, listname, phones)
   print(string.format('| writing %s...', dst))
   os.execute(string.format('mkdir -p %s', dst))

   local idx = 0
   local f
   for filename in io.lines(listname) do
      local target = {}
      local targetfilename = filename:gsub('%.wav$', '%.phn')
      for line in io.lines(src .. "/timit/" .. targetfilename) do
         local s, e, p = line:match('(%S+)%s+(%S+)%s+(%S+)')
         assert(phones[p])
         table.insert(target, p)
      end
      target = table.concat(target, " ")

      idx = idx + 1
      copytoflac(
         src .. "/timit/" .. filename,
         string.format("%s/%09d.flac", dst, idx),
         true
      )
      f = io.open(string.format("%s/%09d.phn", dst, idx), "w")
      f:write(target)
      f:close()
      f = io.open(string.format("%s/%09d.fn", dst, idx), "w")
      f:write(filename)
      f:close()
   end
end

assert(#arg == 2, string.format('usage: %s <src dir> <dst dir>', arg[0]))
local src = arg[1]
local dst = arg[2]

local phones = {}
print(string.format('| analyzing %s...', src .. "/timit/train"))
fileapply(
   src .. "/timit/train",
      '%.phn',
      function(filename)
         for line in io.lines(filename) do
            local s, e, p = line:match('(%S+)%s+(%S+)%s+(%S+)')
            assert(s and e and p)
            if not phones[p] then
               table.insert(phones, p)
               phones[p] = #phones
            end
         end
      end
)
table.sort(
   phones,
   function(a, b)
      return a < b
   end
)
print(string.format('| %d phones found', #phones))
os.execute(string.format('mkdir -p %s', dst))
local f = io.open(dst .. "/phones.lst", "w")
for idx, phone in ipairs(phones) do
   phones[phone] = idx
   f:write(phone .. '\n')
end
f:close()
for idx, phone in ipairs(phones) do
   assert(phones[phone] == idx)
end

createdataset(src, dst .. "/train/", paths.thisfile("train.lst"), phones)
createdataset(src, dst .. "/valid/", paths.thisfile("valid.lst"), phones)
createdataset(src, dst .. "/test/",  paths.thisfile("test.lst"),  phones)

-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- librispeech
-- targets are the transcription strings viewed as ByteTensors

local tnt = require 'torchnet'
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

local function parse_speakers_gender(lines)
   local ret = {}
   for id, g in lines:gmatch("(%d+)[ |]+(%u)[^\n]+[\n]*") do
      ret[tonumber(id)] = g == 'M' and 1 or 0
   end
   return ret
end
print('| parsing SPEAKERS.TXT for gender...')
local gender = parse_speakers_gender(io.open(string.format('%s/SPEAKERS.TXT',
                                             arg[1]), 'r'):read("*all"))

local function fileapply(path, regex, closure)
   local cd = lfs.currentdir()
   local exists = lfs.chdir(path) and true or false
   lfs.chdir(cd)
   if not exists then
      lfs.mkdir(path)
   end
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

local function createidx(src, dst, letters)
   local trans = {}
   assert(type(src) == 'string', 'src path should be a string')
   print(string.format('| analyzing %s...', src))
   fileapply(
      src,
      '%.trans%.txt$',
      function(filename)
         table.insert(trans, filename)
      end
   )
   table.sort(
      trans,
      function(a, b)
         return a < b
      end
   )
   print(string.format('| writing %s...', dst))

   local idx = 0
   for _, filename in pairs(trans) do
      for line in io.lines(filename) do
         local input, lbl = line:match('^(%S+)%s+(.*)$')
         assert(input and lbl)
         idx = idx + 1

         -- wav
         copytoflac(
            string.format('%s%s.flac', filename:gsub('[^/]+$', ''), input),
            string.format('%s/%09d.flac', dst, idx),
            true
         )

         -- words
         lbl = lbl:gsub('^%s+', '')
         lbl = lbl:gsub('%s+$', '')
         lbl = lbl:lower()
         local f = io.open(string.format('%s/%09d.wrd', dst, idx), 'w')
         f:write(lbl)
         f:close()

         -- letters
         lbl = lbl:gsub(' ', '|'):gsub('(.)', '%1 '):gsub(' $', '')
         local f = io.open(string.format('%s/%09d.ltr', dst, idx), 'w')
         f:write(lbl)
         f:close()
         for letter in lbl:gmatch('(%S+)') do
            if not letters[letter] then
               table.insert(letters, letter)
               letters[letter] = #letters
            end
         end

         -- speaker
         local spkr = filename:match('/(%d+)[^/]+$')
         spkr = assert(tonumber(spkr))
         local f = io.open(string.format('%s/%09d.spk', dst, idx), 'w')
         f:write(string.format("%s %s", spkr, gender[spkr]))
         f:close()

         -- filename
         local f = io.open(string.format('%s/%09d.fid', dst, idx), 'w')
         f:write(input)
         f:close()
      end
   end
end

assert(#arg == 2, string.format('usage: %s <src dir> <dst dir>', arg[0]))
local src = arg[1]
local dst = arg[2]

local subpaths = {
   'train-clean-100', 'train-clean-360', 'train-other-500',
   'dev-clean', 'dev-other',
   'test-clean', 'test-other'
}

local letters = {}
-- process subsets
for _, subpath in ipairs(subpaths) do
   local src = string.format("%s/%s", src, subpath)
   local dst = string.format("%s/%s", dst, subpath)
   os.execute(string.format('mkdir -p %s', dst))
   createidx(src, dst, letters)
end

-- save letters list
print('| creating letter list...')
table.sort(
   letters,
   function(a, b)
      return a < b
   end
)
local f = io.open(string.format("%s/letters.lst", dst), "w")
f:write('|' .. '\n')
for i, letter in ipairs(letters) do
   if i < #letters then
      f:write(letter .. '\n')
   end
end
f:close()

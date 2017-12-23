-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- Based on LDC97S62 Switchboard-1 Release 2 only so far
-- Transcriptions are obtained from https://www.isip.piconepress.com/projects/switchboard/ (manually corrected word alignments) - folder name: swb_ms98_transcriptions
-- Adding transcribed versions of other switchboard releases is desirable

local tnt = require 'fbtorchnet'
local sndfile = require 'sndfile'
local xlua = require 'xlua'

require 'lfs'

local function fileapply(path, regex, closure)
   for filename in lfs.dir(path) do
      if filename ~= '.' and filename ~= '..' then
         if lfs.attributes(path .. '/' .. filename, 'mode') == 'directory' then
            fileapply(path .. '/' .. filename, regex, closure)
         else
            if filename:match(regex) then
               closure(path, filename)
            end
         end
      end
   end
end

local dataPaths = {
   "swb1-p1",
   "swb1-p2",
}

local transPaths = {
   "swb_ms98_transcriptions"
}

local function sanitizelbl(lbl)
   --Remove <b_aside> and <e_aside> (background conversation indicators)
   lbl = lbl:gsub('<b_aside>', '')
   lbl = lbl:gsub('<e_aside>', '')
   --Replace [laughter-word] by word (these are words that are still properly understood)
   lbl = lbl:gsub('%[laughter%-([^%]]+)%]', '%1')
   --Treat laughter as silence
   lbl = lbl:gsub('%[laughter%]', '')
   --Choosing first word in [word 1/word 2] scenario (mispronounciation) (noise markings do not use '/')
   lbl = lbl:gsub('%[(%S+)%/%S+%]', '%1')
   --Removing leftover noise markings
   lbl = lbl:gsub('%[[^%[%]]+%]', '')
   --Replacing hyphens by silence (either part of word or from partial pronounciations)
   lbl = lbl:gsub('%-', ' ')
   --Removing capitalizations
   lbl = lbl:lower()
   --Replacing ambersand (&) with ' and ' (used in e.g. h&m)
   lbl = lbl:gsub('&', ' and ')
   --Not removing contractions (')
   --lbl = lbl:gsub('\'', '')
   --Cleaning up white space
   lbl = lbl:gsub('^%s*', '') -- beware
   lbl = lbl:gsub('%s*$', '') -- beware
   lbl = lbl:gsub('%s+', ' ')
   --Remove _%d trailing a word (pronounciation variants)
   lbl = lbl:gsub('_%d', '')
   --Removing brackets {} (made-up words)
   lbl = lbl:gsub('[%{%}/]+', '')

   return lbl
end

local function createProgress(N)
   local n = 0
   return function ()
      n = n + 1
      xlua.progress(n, N)
   end
end

local function createidx(src, dst)
   local data = {}
   for _,p in pairs(transPaths) do
      fileapply(
         src .. '/' .. p,
         'trans.text$',
         function(path, filename)
            local id = filename:sub(1, 6) --Guaranteed to be id by swb manual
            if not data[id] then
               data[id] = {wav = "", a = "", b = ""} --paths to wav file and transcriptions for channel a and b
            end
            local channel = filename:sub(7, 7)
            if channel == 'A' then
               data[id].a = path .. '/' .. filename
            end
            if channel == 'B' then
               data[id].b = path .. '/' .. filename
            end
         end
      )
   end

   for _,p in pairs(dataPaths) do
      fileapply(
         src .. '/' .. p,
         '.wav$',
         function(path, filename)
            local id = filename:sub(1, 6)
            data[id].wav = path .. '/' .. filename
         end
      )
   end


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

   local uttididx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/uttid.idx", dst),
      datafilename = string.format("%s/uttid.bin", dst),
      type = "byte"
   }

   local allSeconds = 0
   local numUtt = 0
   local dataSize = 0
   for _, _ in pairs(data) do dataSize = dataSize + 1 end
   local progress = createProgress(dataSize)
   for id, t in pairs(data) do
      local f = sndfile.SndFile(t.wav)
      local input = f:readFloat(f:info().frames)
      local info = f:info()

      local trackA = input[{{}, {1}}]
      local trackB = input[{{}, {2}}]

      local function process(track, trans, wav)
         local sane = true
         for line in io.lines(trans) do
            local _, from, to, _ = line:match('^(%S+)%s+(%S+)%s+(%S+)%s+(.*)$')
            from = tonumber(from)
            to = tonumber(to)
            -- sanity checks
            if not (from >= 0 and
                    from < info.frames/info.samplerate and
                    to <= info.frames/info.samplerate -- not satisfied if condition above is removed
                    ) then
               sane = false
               break
            end
         end
         for line in io.lines(trans) do
            if not sane then
               print(wav .. ' transcriptions fail assertions -> ignored')
               break
            end
            local uttid, from, to, lbl = line:match('^(%S+)%s+(%S+)%s+(%S+)%s+(.*)$')
            from = tonumber(from)
            to = tonumber(to)
            local size = (to - from) * info.samplerate

            if not uttid or not from or not to or not lbl then
               error(string.format('could not parse properly file %s', trans))
            end

            --TODO: Replace numbers by their spelling?
            lbl = sanitizelbl(lbl)

            -- ignore if transcriptions contains numbers
            -- -- numbers are only used in cases where they have a specific meaning, so it's not clear what is actually being said
            -- ignore if transcription contains only silence
            if not lbl:match('^%s$') and not lbl:match('[0-9]') and lbl ~= '' and size > 1 then
               if lbl:match('[^a-z%s\']') then
                  print('nonstd ' .. uttid .. ' - ' .. lbl)
               end

               local tmpfname = '/dev/shm/3189913856429815649.wav'
               local out = sndfile.SndFile(tmpfname, 'w', {samplerate=info.samplerate, channels=1, format="WAV", subformat="ULAW"})
               local start = from * info.samplerate
               local input = track:narrow(1, start + 1, size)
               out:writeFloat(input)
               out:close()

               local fin = torch.DiskFile(tmpfname)
               fin:seekEnd()
               local size = fin:position()
               fin:seek(1)
               local stor = fin:readByte(size-1)
               fin:close()

               inputidx:add(torch.ByteTensor(stor)) --specifying the filename appears to ask for a 'char'-typed indexeddataset
               targetidx:add(torch.ByteTensor(torch.ByteStorage():string(lbl)))
               uttididx:add(torch.ByteTensor(torch.ByteStorage():string(uttid)))

               allSeconds = allSeconds + size / info.samplerate
               numUtt = numUtt + 1
            end
         end
      end
      progress()
      process(trackA, t.a, t.wav)
      process(trackB, t.b, t.wav)
   end
   print(string.format("%d hours stored", allSeconds / (60*60)))
   print(string.format("%d utterances stored", numUtt))

   inputidx:close()
   targetidx:close()
   uttididx:close()
end

assert(#arg == 2, string.format('usage: %s <src dir> <dst dir>', arg[0]))
local src = arg[1]
local dst = arg[2]

os.execute(string.format('mkdir -p %s', dst))
createidx(src, dst)

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

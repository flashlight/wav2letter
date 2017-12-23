-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- Fisher consists of the LDC datasets
-- LDC2005T19 - Fisher English Training Part 2, Transcripts
-- LDC2005S13 - Fisher English Training Part 2, Speech
-- LDC2004T19 - Fisher English Training Speech Part 1 Transcripts
-- LDC2004S13 - Fisher English Training Speech Part 1 Speech
-- There is no explicit evaluation or test data yet
-- I am assuming, that Part 2 and Part 1 transcripts follow the same guidelines!

local paths = require 'paths'
local sndfile = require 'sndfile'
local xlua = require 'xlua'
require 'lfs'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Fisher Dataset creation')
cmd:text()
cmd:argument('-part1', 'top level directory containing all part1 discs')
cmd:argument('-part2', 'top level directory containing all part2 discs')
cmd:text()
cmd:text('Options:')
cmd:option('-dst', "./fisher", "destination directory")
cmd:option('-sph2pipe', "./sph2pipe_v2.5/sph2pipe", "path to sph2pipe executable")
cmd:option('-progress', false, "show progress bar")
cmd:option('-noaudio', false, "no do look at audio files")
cmd:text()

local opt = cmd:parse(arg)

local src = {opt.part1, opt.part2}
local dst = opt.dst
local sph2pipe = opt.sph2pipe

local f = io.open(sph2pipe)
if not f then
   error(string.format('sph2pipe not found at <%s> -- please provide a correct path', sph2pipe))
end
f:close()

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

local subparts = {
   "part1",
   "part2",
}

local dataPaths1 = {
   "fisher_eng_tr_sp_d1",
   "fisher_eng_tr_sp_d2",
   "fisher_eng_tr_sp_d3",
   "fisher_eng_tr_sp_d4",
   "fisher_eng_tr_sp_d5",
   "fisher_eng_tr_sp_d6",
   "fisher_eng_tr_sp_d7",
}

local dataPaths2 = {
   "fe_03_p2_sph1/",
   "fe_03_p2_sph2/",
   "fe_03_p2_sph3/",
   "fe_03_p2_sph4/",
   "fe_03_p2_sph5/",
   "fe_03_p2_sph6/",
   "fe_03_p2_sph7/",
}

local function createProgress(N)
   local n = 0
   return function ()
      n = n + 1
      xlua.progress(n, N)
   end
end

local function fsize (file)
   local current = file:seek()      -- get current position
   local size = file:seek("end")    -- get file size
   file:seek("set", current)        -- restore position
   return size
end

local function createidx()
   local trans = {}
   for _, p in pairs(src) do
      fileapply(
         p,
         'fe_03_[%d]+.txt',
         function(filename)
            table.insert(trans, filename)
         end
      )
   end
   print(string.format("| Found %d transcriptions", #trans))

   local data = {}
   local datac = 0
   for _, p in pairs(src) do
      fileapply(
         p,
         '.sph',
         function(filename)
            local uttid = paths.basename(filename):sub(1, 11)
            data[uttid] = filename
            datac = datac + 1
         end
      )
   end
   print(string.format("| Found %d data files", datac))

   local allSeconds = 0
   local uttCount = 0

   local progress = createProgress(#trans)
   for _, filename in pairs(trans) do
      local transText = io.open(filename)
      local line1 = transText:read()
      local uttid = line1:sub(3, 13)
      local wav = data[uttid]
      local _ = transText:read()
      if not wav then
         print('Wav not found for transcription: ' .. filename .. ' with uttid: ' .. uttid)
      end
      if wav and line1:match('# fe_') then

         local track
         local info
         if not opt.noaudio then
            -- sph to wav
            local f = torch.PipeFile(string.format('%s -f wav -p "%s"', sph2pipe, wav)):quiet()
            local wav = f:readByte(2^30)
            assert(wav:size() > 0)
            f:close()

            -- reading wav
            local f = sndfile.SndFile(wav)
            info = f:info()
            track = f:readShort(info.frames)
            f:close()
         end

         while true do
            local line = transText:read('*l')
            if line  == nil then break end
            if not line:match('^#') and line:match('%S') then
               --Getting start, end, channel and text
               local s, e, c, words = line:match('^(%S+)%s+(%S+)%s+(%S):%s+(.*)$')
               s = tonumber(s)
               e = tonumber(e)
               assert(words)
               local channel
               if c == 'A' then
                  channel = 1
               else
                  channel = 2
                  assert(c == 'B')
               end

               --Processing label
               words = words:lower()

               -- remove punctuation
               words = words:gsub(',', '')
               words = words:gsub('?', '')

               -- simplify noise annotations
               words = words:gsub("%[%[skip%]%]", "")
               words = words:gsub("%[pause%]", "")
               words = words:gsub("%[laugh%]", "[laughter]")
               words = words:gsub("%[sigh%]", "[noise]")
               words = words:gsub("%[cough%]", "[noise]")
               words = words:gsub("%[mn%]", "[noise]")
               words = words:gsub("%[breath%]", "[noise]")
               words = words:gsub("%[lipsmack%]", "[noise]")
               words = words:gsub("%[sneeze%]", "[noise]")

               -- strip
               words = words:gsub('%s+', ' ')
               words = words:gsub('^%s+', '')
               words = words:gsub('%s+$', '')

               local spellings = words:gsub('%[laughter%]', 'L')
               spellings = spellings:gsub('%[noise%]', 'N')
               spellings = spellings:gsub("401k", "four-o-one-k")
               spellings = spellings:gsub("ak%-47", "ak-forty-seven")
               spellings = spellings:gsub("ak47", "ak-forty-seven")
               spellings = spellings:gsub("u2", "u-two")
               spellings = spellings:gsub("v8", "v-eight")
               spellings = spellings:gsub("mp3", "m-p-three")
               spellings = spellings:gsub("m16", "m-sixteen")
               spellings = spellings:gsub("f16", "f-sixteen")
               spellings = spellings:gsub("dc3", "dc-three")
               spellings = spellings:gsub("y2k", "y-two-k")
               spellings = spellings:gsub("3d", "three-d")
               spellings = spellings:gsub("espn2", "e-s-p-n-two")
               spellings = spellings:gsub("vh1", "vh-one")
               spellings = spellings:gsub("s2b", "s-two-b")
               spellings = spellings:gsub("90210", "nine-o-two-one-o")
               spellings = spellings:gsub("2", "two")
               spellings = spellings:gsub('%&', '-n-')
               spellings = spellings:gsub('_', '')
               spellings = spellings:gsub('-', '')
               spellings = spellings:gsub('%.', '')
               spellings = spellings:gsub('%*', '')
               spellings = spellings:gsub('%~', '')
               spellings = spellings:gsub('%s+', ' ')
               spellings = spellings:gsub('^%s+', '')
               spellings = spellings:gsub('%s+$', '')
               spellings = spellings:gsub('%s', '|'):gsub('(.)', '%1 '):gsub(' $', '')

               local start
               local size
               if not opt.noaudio then
                  start = s * info.samplerate
                  size = math.min((e - s) * info.samplerate, info.frames-start)
               end

               local sane = true
               -- remove uncertain annotations
               if words:match('%(%(') or words == "" then
                  sane = false
               end

               if sane and spellings:match('[^a-z%s\'|LN]') then
                  error("Improperly processed " .. uttid)
               end

               if sane then
                  local subdirid = math.floor(uttCount / 10000)
                  local subdir = paths.concat(dst, string.format('%05d', subdirid))
                  if uttCount % 10000 == 0 then
                     os.execute(string.format('mkdir -p %s', subdir))
                  end

                  if not opt.noaudio then
                     local input = track:narrow(1, start + 1, size):narrow(2, channel, 1)
                     allSeconds = allSeconds + (size / info.samplerate)

                     -- wav
                     local f = sndfile.SndFile(string.format('%s/%09d.flac', subdir, uttCount), 'w', {samplerate=info.samplerate, channels=1, format='FLAC', subformat="PCM16"})
                     f:writeShort(input)
                     f:close()
                  end

                  -- words
                  local f = io.open(string.format('%s/%09d.wrd', subdir, uttCount), 'w')
                  f:write(words)
                  f:close()

                  -- letters
                  local f = io.open(string.format('%s/%09d.ltr', subdir, uttCount), 'w')
                  f:write(spellings)
                  f:close()

                  -- uid (utterance id + original time stamps)
                  local f = io.open(string.format('%s/%09d.uid', subdir, uttCount), 'w')
                  f:write(string.format("%s %s %s", uttid, s, e))
                  f:close()

                  uttCount = uttCount + 1
               end
            end
         end
      end
      if opt.progress then
         progress()
      end
   end

   print(string.format("Written %d hours of speech", allSeconds / (60 * 60)))
   print(string.format("Written %d utterances", uttCount))
end

os.execute(string.format('mkdir -p %s', dst))
createidx()

-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- wsj datasets
-- creating common data sets
-- please install sph2pipe on your own
-- see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools

local tnt = require 'torchnet'
local paths = require 'paths'
local xlua = require 'xlua'
local sndfile = require 'sndfile'

torch.setheaptracking(true)


local cmd = torch.CmdLine()
cmd:text()
cmd:text('WSJ Dataset creation')
cmd:text()
cmd:argument('-WSJ0', 'top level directory containing all WSJ0 discs')
cmd:argument('-WSJ1', 'top level directory containing all WSJ1 discs')
cmd:text()
cmd:text('Options:')
cmd:option('-dst', "./wsj", "destination directory")
cmd:option('-sph2pipe', "./sph2pipe_v2.5/sph2pipe", "path to sph2pipe executable")
cmd:text()

local opt = cmd:parse(arg)

local WSJ0 = opt.WSJ0
local WSJ1 = opt.WSJ1
local dst  = opt.dst
local sph2pipe  = opt.sph2pipe
local sets = {}

local f = io.open(sph2pipe)
if not f then
   error(string.format('sph2pipe not found at <%s> -- please provide a correct path', sph2pipe))
end
f:close()

local preprocess = paths.dofile("preprocess.lua")

-- find transcripts first
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

local transcripts = {}
local function findtrans(filename)
   local subset = filename:match('([^/]+)/[^/]+/[^/]+$')
   transcripts[subset] = transcripts[subset] or {}
   assert(subset)
   for line in io.lines(filename) do
      local trans, id = line:match('^(.+)%s+%((%S+)%)%s*$')
      if trans and id then
         if transcripts[subset][id] and transcripts[subset][id] ~= trans then
            error(string.format("different transcriptions available for #%s:\n%s\n%s", id, trans, transcripts[id]))
         end
         transcripts[subset][id] = trans
      end
   end
end

fileapply(
   WSJ0,
   '.*%.dot',
   findtrans
)

fileapply(
   WSJ1,
   '.*%.dot',
   findtrans
)

-- now read file list for each dataset
local function ndx2idlist(prefix, filename, transform, list)
   list = list or {}
   for line in io.lines(paths.concat(prefix, filename)) do
      if transform then
         line = transform(line)
      end
      if line and not line:match('^%;') then
         local id = line:match('([^/]+)$')
         id = id:gsub('%.wv1', '')
         local p1, p2, p3, filename = line:match('^(%d+)_(%d+)_(%d+)%:[%s/]*(.+)$')
         assert(p1 and p2 and p3 and filename, string.format("could not parse line <%s>", line))
         local sep = line:match('wsj0') and '-' or '_'
         local subset = line:match('([^/]+)/[^/]+/[^/]+$')
         assert(subset)
         local trans = transcripts[subset][id]
         if not trans then
            error(string.format("transcript not found for subset=<%s> id=<%s>", subset, id))
         end
         filename = paths.concat(prefix, string.format("%d%s%d.%d", p1, sep, p2, p3), filename)
         local f = io.open(filename)
         assert(f, string.format("<%s> not found", filename))
         f:close()
         table.insert(list, {id=id, filename=filename, subset=subset, transcript=trans})
      end
   end
   table.sort(
      list,
      function(a, b)
         return a.id < b.id
      end
   )
   collectgarbage()
   print(string.format("# Read %d samples (and transcriptions) in <%s>", #list, filename))
   return list
end

sets.si84 = ndx2idlist(
   WSJ0,
   '11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx',
   function(line)
      if line:match('11_2_1:wsj0/si_tr_s/401') then
         return nil
      else
         return line
      end
   end
)
assert(#sets.si84 == 7138)

sets.si284 = ndx2idlist(
   WSJ0,
   '11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx',
   function(line)
      if line:match('11_2_1:wsj0/si_tr_s/401') then
         return nil
      else
         return line
      end
   end
)

sets.si284 = ndx2idlist(
   WSJ1,
   '13_34.1/wsj1/doc/indices/si_tr_s.ndx',
   nil,
   sets.si284
)
assert(#sets.si284 == 37416)

sets.nov92 = ndx2idlist(
   WSJ0,
   '11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx',
   function(line)
      return line .. '.wv1'
   end
)
assert(#sets.nov92 == 333)

sets.nov92_5k = ndx2idlist(
   WSJ0,
   '11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx',
   function(line)
      return line .. '.wv1'
   end
)
assert(#sets.nov92_5k == 330)

sets.nov93 = ndx2idlist(
   WSJ1,
   '13_32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx',
   function(line)
      line = line:gsub('13_32_1', '13_33_1')
      return line
   end
)
assert(#sets.nov93 == 213)

sets.nov93_5k = ndx2idlist(
   WSJ1,
   '13_32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx',
   function(line)
      line = line:gsub('13_32_1', '13_33_1')
      return line
   end
)
assert(#sets.nov93_5k == 215)

sets.nov93dev = ndx2idlist(
   WSJ1,
   '13_34.1/wsj1/doc/indices/h1_p0.ndx'
)
assert(#sets.nov93dev == 503)

sets.nov93dev_5k = ndx2idlist(
   WSJ1,
   '13_34.1/wsj1/doc/indices/h2_p0.ndx'
)
assert(#sets.nov93dev_5k == 513)

local function transcript2wordspelling(transcript, filename)
   local words = {}
   local spellings = {}
   for token in transcript:gmatch('(%S+)') do
      local word, spelling = preprocess(token)
      if word and spelling then
         if word:match("[^abcdefghijklmnopqrstuvwxyz'#]") then
            error("invalid transcript <%s> for filename <%s>", transcript, filename)
         end
         table.insert(words, word)
         table.insert(spellings, spelling)
      end
   end
   return table.concat(words, ' '), table.concat(spellings, ' ')
end

for name, list in pairs(sets) do
   print(string.format("# Writing %s with %d samples", name, #list))
   local dst = string.format("%s/%s", dst, name)
   os.execute(string.format('mkdir -p "%s"', dst))
   for idx, sample in ipairs(list) do
      local filename = sample.filename
      local id = sample.id
      local subset = sample.subset
      local words, spellings = transcript2wordspelling(sample.transcript, filename)
      local f = torch.PipeFile(string.format('%s -f wav "%s"', sph2pipe, filename)):quiet()
      local wav = f:readByte(2^30)
      assert(wav:size() > 0)
      f:close()

      -- wav
      local f = sndfile.SndFile(wav)
      local info = f:info()
      local data = f:readShort(info.frames)
      f:close()
      local f = sndfile.SndFile(string.format('%s/%09d.flac', dst, idx), 'w', {samplerate=info.samplerate, channels=info.channels, format='FLAC', subformat="PCM16"})
      f:writeShort(data)
      f:close()

      -- words
      local f = io.open(string.format('%s/%09d.wrd', dst, idx), 'w')
      f:write(words)
      f:close()

      -- letters
      local f = io.open(string.format('%s/%09d.ltr', dst, idx), 'w')
      spellings = spellings:gsub('%s', '|'):gsub('(.)', '%1 '):gsub(' $', '')
      f:write(spellings)
      f:close()

      -- id
      local f = io.open(string.format('%s/%09d.fid', dst, idx), 'w')
      f:write(id)
      f:close()

      -- filename
      local f = io.open(string.format('%s/%09d.fln', dst, idx), 'w')
      f:write(filename)
      f:close()

      xlua.progress(idx, #list)
   end
end

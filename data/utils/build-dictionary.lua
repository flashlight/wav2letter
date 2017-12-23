-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'wav2letter'

local tnt = require 'torchnet'
local readers = require 'wav2letter.readers'
local spellingremoveletterrepeat = paths.dofile('spelling.lua')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Extract/Augment a word dictionary with unknown words from a corpus')
cmd:text()
cmd:argument('corpora', 'corpora separated by "+"')
cmd:text()
cmd:text('Options:')
cmd:option('-datadir', string.format('%s/local/datasets/speech', os.getenv('HOME')), 'speech data directory')
cmd:option('-r', 2, 'repetition letters')
cmd:option('-srcdict', '', 'src dictionary')
cmd:option('-dstdict', '', 'dst dictionary')
cmd:option('-unkdst', false, 'dst dictionary contains only unknowns')
cmd:text()

local opt = cmd:parse(arg)

local dict = {}
local unk = {}

local f = io.stdout
local function writedict(word, spelling)
   spelling = spelling or spellingremoveletterrepeat(word, opt.r)
   f:write(string.format('%s %s\n', word, spelling))
end

if opt.dstdict ~= '' then
   f = io.open(opt.dstdict, 'w')
   assert(f, 'could not open dstdict')
end

if opt.srcdict ~= '' then
   for line in io.lines(opt.srcdict) do
      local word, spelling = line:match('^(%S+)%s+(%S+)$')
      if not opt.unkdst then
         writedict(word, spelling)
      end
      assert(word and spelling)
      dict[word] = spelling
   end
end

for path in opt.corpora:gmatch('([^%+]+)') do
   path = paths.concat(opt.datadir, path)
   local dataset = tnt.NumberedFilesDataset{
      path = path,
      features = {
         {
            name = "wrd",
            alias = "words",
            reader = readers.words()
         }
      }
   }
   for i=1,dataset:size() do
      local words = dataset:get(i).words
      for word in words:gmatch('(%S+)') do
         if not dict[word] then
            if not unk[word] then
               writedict(word, spellingremoveletterrepeat(word, 2))
               unk[word] = true
            end
         end
      end
   end
end

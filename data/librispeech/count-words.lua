-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'paths'

local root = arg[1]

assert(root, '<path to librispeech-idx> expected as argument')

local tnt = require 'fbtorchnet'

local function countwords(arg)
   local words = {}

   for _, dir in ipairs(arg) do
      local data = tnt.IndexedDatasetReader{
         indexfilename = string.format("%s/target.idx", dir),
         datafilename  = string.format("%s/target.bin", dir),
         mmap = true,
         mmapidx = true,
      }

      for i=1,data:size() do
         local sentence = data:get(i):clone():storage():string()
         for word in sentence:gmatch('(%S+)') do
            words[word] = words[word] or 0
            words[word] = words[word] + 1
         end
      end
   end

   local sortedwords = {}
   for word, count in pairs(words) do
      table.insert(sortedwords, {word=word, count=count})
   end
   table.sort(
      sortedwords,
      function(a, b)
         return a.count > b.count
      end
   )

   local total = 0
   for i=1,#sortedwords do
      local s = sortedwords[i]
      total = total + s.count
   end

   for i=1,#sortedwords do
      local s = sortedwords[i]
      s.freq = s.count/total*100
      sortedwords[s.word] = s
   end

   return sortedwords
end

local train = countwords{
   paths.concat(root, "train-clean-100"),
   paths.concat(root, "train-clean-360"),
   paths.concat(root, "train-other-500")
}
local test  = countwords{
   paths.concat(root, "test-clean"),
   paths.concat(root, "test-other")
}

local f = io.open(paths.concat(root, 'words.lst'), 'w')
local traincumprob = 0
local testcumprob = 0
for i=1,#train do
   local s = train[i]
   local trainfreq = s.freq
   local testfreq = test[s.word] and test[s.word].freq or 0
   traincumprob = traincumprob + trainfreq
   testcumprob = testcumprob + testfreq
   f:write(string.format("%s rank: %d count: %d train freq: %03.2f%% test freq: %03.2f%% train cumprob: %03.2f%% test cumprob: %03.2f%%\n", s.word, i, s.count, trainfreq, testfreq, traincumprob, testcumprob))
end
f:close()

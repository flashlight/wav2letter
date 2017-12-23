-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'torch'

local beamer = require 'beamer'

torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'
paths.dofile('wav2letter/fullconnectcriterion.lua')

local function loadhash(filename, maxn)
   print(string.format('[loading %s]', filename))
   local hash = {}
   local i = 0
   for word in io.lines(filename) do
      word = word:match('^(%S+)') or word
      hash[word] = i -- 0 based
      hash[i] = word
      i = i + 1
      if maxn and maxn > 0 and maxn == i then
         break
      end
   end
   print(string.format('[%d tokens found]', i))
   return hash
end

local words = loadhash('vocab.lst', 90000)
local letters = loadhash('letters.lst')

local buffer = torch.LongTensor()
local function word2indices(word)
   buffer:resize(#word)
   for i=1,#word do
      if not letters[word:sub(i, i)] then
         error(string.format('unknown letter <%s>', word:sub(i, i)))
      end
      buffer[i] = letters[word:sub(i, i)]
   end
   return buffer
end

local lm = beamer.LM("lm-uniq-90k.bin")

if false then
   local sentence = "the cat sat on the mat"
   local state = lm:start()
   local totalscore = 0
   local score
   print(string.format('%s\tscore = %.2f\ttotalscore = %.2f', '<s>', 0, totalscore))
   for word in sentence:gmatch('(%S+)') do
      state, score = lm:score(state, lm:index(word))
      totalscore = totalscore + score
      print(string.format('%s\tscore = %.2f\ttotalscore = %.2f', word, score, totalscore))
   end
   state, score = lm:finish(state)
   totalscore = totalscore + score
   print(string.format('%s\tscore = %.2f\ttotalscore = %.2f', '</s>', score, totalscore))
end

local trie = beamer.Trie(#letters+1)
local lmidx2word = {}
for i=0,#words do
   local lmidx = lm:index(words[i])
   if lmidx > 0 then
      trie:insert(word2indices(words[i]), lmidx)
      lmidx2word[lmidx] = words[i]
   end
end

if false then
   for i=0,#words do
      local word = words[i]
      for i=1,#word do
         local word = word:sub(1, -i)
         local buffer = word2indices(word)
         local idx = trie:search(buffer):label()
         if words[word] then
            if idx < 0 then
               error(string.format('unknown word <%s> (words idx = %d, trie idx = %d)', word, words[word], idx))
            end
            assert(lmidx2word[idx] == word)
         else
            assert(idx < 0)
         end
      end
   end
end

local sil = letters[' ']
local f = torch.DiskFile('netout.bin'):binary()
local transitions = f:readObject()
local emissions = f:readObject()
f:close()
print('transitions', transitions:size(), transitions:isContiguous())
print('emissions', emissions:size(), emissions:isContiguous())

local criterion = nn.FullConnectCriterion(emissions:size(2))
criterion.transitions:copy(transitions)
local res,score = criterion:viterbi(emissions)
local sentence = {}
res:apply(
   function(idx)
      assert(letters[idx])
      table.insert(sentence, letters[idx-1])
   end
)
print(score)
print(table.concat(sentence))

local beam = 1000
local decoder = beamer.Decoder(trie, lm, 0.9, beam, sil)
local scores = torch.FloatTensor()
local labels = torch.LongTensor()
local llabels = torch.LongTensor()

for Z=1,2 do
   local timer = torch.Timer()
   decoder:decode(transitions, emissions, scores, llabels, labels)
   print(string.format('[number of final hypothesis: %d]', scores:size(1)))

   print(scores:narrow(1, 1, math.min(5, scores:size(1))))

   for i=1,math.min(3, labels:size(1)) do
      local sentence = {}
      local labels = labels[i]
      local llabels = llabels[i]
      for j=1,labels:size(1) do
         local letteridx = llabels[j]
         local wordidx = labels[j]
         if letteridx >= 0 then
            assert(letters[letteridx])
            table.insert(sentence, letters[letteridx])
         else
            table.insert(sentence, '*')
         end
         if wordidx >= 0 then
            assert(words[wordidx])
            --DEBUG:         table.insert(sentence, string.format('[%s]', words[wordidx]))
            table.insert(sentence, string.format('[%s]', lmidx2word[wordidx]))
         end
      end
      print(table.concat(sentence, ''))
   end
   print(string.format('[real time: %.2f s]', timer:time().real))
   print(string.format('[lexicon trie memory %.2f Mb]', trie:mem()/2^20))
   print(string.format('[lm state memory %.2f Mb]', lm:mem()/2^20))
   print(string.format('[decoder state memory %.2f Mb]', decoder:mem()/2^20))
end

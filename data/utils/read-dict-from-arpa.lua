-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local f = io.open(arg[1])

local unigram
repeat
   local line = f:read('*line')
   local n = tonumber(line:match('ngram%s+1%=(%S+)'))
   if n then
      unigram = n
      io.stderr:write('#unigram ' .. unigram .. '\n')
      io.stderr:flush()
   end
   if line:match('\\1%-grams%:') then
      break
   end
until not line

local words = {}
for i=1,unigram do
   local line = f:read('*line')
   local word = line:match('%S+%s+(%S+)')
   if words[word] then
      error('ouch')
   end
   if not word:match('<') then
      words[word] = true
   end
end

local sortedwords = {}
for word, _ in pairs(words) do
   table.insert(sortedwords, word)
end

table.sort(
   sortedwords,
   function(a, b)
      return a < b
   end
)

for _, word in ipairs(sortedwords) do
   print(word)
end

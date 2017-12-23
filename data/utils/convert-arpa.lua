-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'torch'
local spellingremoveletterrepeat = paths.dofile('spelling.lua')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Cleanup existing ARPA language model for wav2letter')
cmd:text()
cmd:argument('-arpasrc', 'src ARPA language model (can be zipped)')
cmd:argument('-arpadst', 'dst ARPA language model')
cmd:argument('-dictdst', 'dst dictionary')
cmd:text()
cmd:text('Options:')
cmd:option('-preprocess', '', "a lua file which returns a word pre-processing function, if any")
cmd:option('-r', 0, "max number of letter repetitions allowed - if 0, do nothing")
cmd:option('-letters', '', "a letter dictionary (only to check if word spellings are valid)")
cmd:text()

local opt = cmd:parse(arg)

-- read letters if provided
local letters
if opt.letters ~= '' then
   letters = {}
   for letter in io.lines(opt.letters) do
      letters[letter] = true
   end
end

-- basic word check
local function checkspelling(spelling, letters)
   if spelling ~= '<s>' and spelling ~= '</s>' and spelling ~= '<unk>' then
      for i=1,#spelling do
         local letter = spelling:sub(i, i)
         if not letters[letter] then
            error('invalid spelling <' .. spelling .. '> (letter not in dictionary)')
         end
      end
   end
   return spelling
end

-- read pre-processing function, if any
local preprocess = function(word) return word end
if opt.preprocess ~= '' then
   preprocess = dofile(opt.preprocess)
end

local data = false -- are we in data section?
local ngrams = {} -- ngram count
local dict = {} -- dst word dictionary
local n -- current ngram number
local nmask -- current ngram string match
local nmaskbackoff -- current ngram string match (with backoff)
local ofarpa = io.open(opt.arpadst, 'w')
assert(ofarpa, string.format('cannot open <%s> for writing', opt.arpadst))

-- src arpa
local ifarpa
if opt.arpasrc:match('%.z$') or opt.arpasrc:match('%.gz$') then
   ifarpa = io.popen(string.format('zcat %s', opt.arpasrc))
else
   ifarpa = io.open(opt.arpasrc)
end

for line in ifarpa:lines() do
   if line:match('^\\data\\$') then
      data = true
      ofarpa:write(line .. '\n')
   end

   if n and line:match('^\\end\\$') then
      ofarpa:write(line .. '\n')
      break
   end

   if data then
      local n, num = line:match('ngram (%d)%=(%d+)')
      n = tonumber(n)
      num = tonumber(num)
      if n and num then
         ngrams[n] = num
         ofarpa:write(line .. '\n')
      end
   end

   if (data or n) and line:match('^\\%d%-grams:$') then
      ofarpa:write('\n')
      n = tonumber(line:match('^\\(%d)%-grams:$'))
      data = nil
      nmask = '(%S+)' .. string.rep('%s+(%S+)', n)
      nmaskbackoff = '(%S+)' .. string.rep('%s+(%S+)', n+1)
      ofarpa:write(line .. '\n')
   end

   if n then
      local res = {line:match(nmaskbackoff)}
      local valid = false
      if #res == n+2 and tonumber(res[1]) and tonumber(res[#res]) then
         valid = true
      else
         res = {line:match(nmask)}
         if #res == n+1 and tonumber(res[1]) then
            valid = true
         end
      end
      if valid then
         for i=1,n do
            local word = res[i+1]
            if word == '<UNK>' or word == '<unk>' then
               res[i+1] = '<unk>'
            elseif word ~= '<s>' and word ~= '</s>' then
               local word, spelling = preprocess(word)
               assert(word, 'preprocess() is supposed to return <word[, spelling]>')
               spelling = spelling or word
               if letters then
                  spelling = checkspelling(spelling, letters)
               end
               if opt.r > 0 then
                  spelling = spellingremoveletterrepeat(spelling, opt.r)
               end
               res[i+1] = word
               if n == 1 then
                  if dict[word] then
                     print(string.format("! warning: word <%s> duplicated in the dictionary", word))
                  else
                     dict[word] = true
                     table.insert(dict, {word=word, spelling=spelling})
                  end
               end
            end
         end
         ofarpa:write(table.concat(res, '\t') .. '\n')
      end
   end
end
ofarpa:close()

table.sort(
   dict,
   function(a, b)
      return a.word < b.word
   end
)
local f = io.open(opt.dictdst, 'w')
for _, entry in ipairs(dict) do
   f:write(string.format("%s\t%s\n", entry.word, entry.spelling))
end
f:close()

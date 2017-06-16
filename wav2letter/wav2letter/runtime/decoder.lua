local beamer = require 'beamer'

local LMUNK = "<unk>"

local function loadwordhash(filename, maxn)
   print(string.format('[loading %s]', filename))
   local hash = {}
   local i = 0 -- 0 based
   for line in io.lines(filename) do
      local word, spelling = line:match('^(%S+)%s+(%S+)$')
      assert(word and spelling, string.format("error parsing <%s> at line #%d", filename, i+1))
      if not hash[word] then
         hash[word] = {idx=i, word=word, spellings={}}
         hash[i] = hash[word]
         i = i + 1
      end
      table.insert(hash[word].spellings, spelling)
      if maxn and maxn > 0 and maxn == i then
         break
      end
   end
   print(string.format('[%d tokens found]', i))
   return hash
end

local function loadletterhash(filename, maxn)
   print(string.format('[loading %s]', filename))
   local hash = {}
   local i = 0 -- 0 based
   for letter in io.lines(filename) do
      letter = letter:match('([^\n]+)') or letter
      hash[letter] = i
      hash[i] = letter
      i = i + 1
   end
   print(string.format('[%d letters found]', i))
   return hash
end


-- config and opt options?
local function decoder(letterdictname, worddictname, lmname, smearing, nword)
   local words = loadwordhash(worddictname, nword)
   local letters = loadletterhash(letterdictname)

   -- add <unk> in words (only :))
   if not words[LMUNK] then
      local def = {idx=#words+1, word=LMUNK, spellings={}}
      words[def.idx] = def
      words[def.word] = def
   end

   local buffer = torch.LongTensor()
   local function spelling2indices(word)
      buffer:resize(#word)
      for i=1,#word do
         if not letters[word:sub(i, i)] then
            error(string.format('unknown letter <%s>', word:sub(i, i)))
         end
         buffer[i] = letters[word:sub(i, i)]
      end
      return buffer
   end

   local lm = beamer.LM(lmname)
   local sil = letters['|']
   local unk = {lm=lm:index(LMUNK), usr=words[LMUNK].idx}

   local trie = beamer.Trie(#letters+1, sil) -- 0 based
   for i=0,#words do
      local lmidx = lm:index(words[i].word)
      local _, score = lm:score(lmidx)
      assert(score < 0)
      for _, spelling in ipairs(words[i].spellings) do
         trie:insert(spelling2indices(spelling), {lm=lmidx, usr=i}, score)
      end
   end

   local function toword(usridx)
      local word = words[usridx]
      assert(word, 'unknown word index')
      return word.word
   end

   if smearing == 'max' then
      trie:smearing()
   elseif smearing == 'logadd' then
      trie:smearing(true)
   elseif smearing ~= 'none' then
      error('smearing should be none, max or logadd')
   end

   print(string.format('[Lexicon Trie memory usage: %.2f Mb]', trie:mem()/2^20))
   local decoder = beamer.Decoder(trie, lm, sil, unk)
   decoder:settoword(toword)
   local scores = torch.FloatTensor()
   local labels = torch.LongTensor()
   local llabels = torch.LongTensor()

   local function decode(opt, transitions, emissions, K)
      K = K or 1
      decoder:decode(opt, transitions, emissions, scores, llabels, labels)
      local function bestpath(k)
         local sentence = {}
         local lsentence = {}
         if labels:nDimension() > 0 then
            local labels = labels[k]
            local llabels = llabels[k]
            for j=1,labels:size(1) do
               local letteridx = llabels[j]
               local wordidx = labels[j]
               if letteridx >= 0 then
                  assert(letters[letteridx])
                  table.insert(lsentence, letteridx)
               end
               if wordidx >= 0 then
                  assert(words[wordidx])
                  table.insert(sentence, wordidx)
               end
            end
         end
         return torch.LongTensor(sentence), torch.LongTensor(lsentence), scores[k]
      end
      if K == 1 then
         return bestpath(1)
      else
         local sentences = {}
         local lsentences = {}
         local scores = {}
         for k=1,K do
            local sentence, lsentence, score = bestpath(k)
            table.insert(sentences, sentence)
            table.insert(lsentences, lsentence)
            table.insert(scores, score)
         end
         return sentences, lsentences, scores
      end
   end

   local obj = {
      words = words,
      letters = letters,
      lm = lm,
      trie = trie,
      sil = sil,
      decoder = decoder,
      decode = decode,
      toword = toword, -- usridx to word
      spelling2indices = spelling2indices -- word to letter idx
   }
   setmetatable(obj, {__call=function(...) return decode(select(2, ...), select(3, ...)) end})

   return obj
end

return decoder

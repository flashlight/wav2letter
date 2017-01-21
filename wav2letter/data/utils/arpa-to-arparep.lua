require 'torch'

local header = false
local ngram
local format

local cmd = torch.CmdLine()
cmd:text()
cmd:text('SpeechRec (c) Ronan Collobert 2015')
cmd:text()
cmd:text('Arguments:')
cmd:argument('-letters', 'letter dictionary (lower case) to check if words are valid')
cmd:argument('-arpa', 'original arpa file')
cmd:argument('-arparep', 'arpa with rep (out)')
cmd:text()

local opt = cmd:parse(arg)

local letters = {}
for letter in io.lines(opt.letters) do
   letters[letter] = true
end

local function checkword(word)
   word = word:lower()
   if word:match('%d') then
      error('invalid word <' .. word .. '> (no number allowed)')
   end
   if word ~= '<s>' and word ~= '</s>' and word ~= '<unk>' then
      for i=1,#word do
         local letter = word:sub(i, i)
         if not letters[letter] then
            error('invalid word <' .. word .. '> (letter not in dictionary)')
         end
      end
   end
   local prevletter = word:sub(1, 1)
   local newword = prevletter
   local r = 1
   local invalid = false
   for i=2,#word do
      local letter = word:sub(i, i)
      if letter == prevletter then
         r = r + 1
         if r > 3 then
            r = 1
            newword = newword .. letter
            invalid = true
         else
            newword = newword .. r
         end
      else
         r = 1
         prevletter = letter
         newword = newword .. letter
      end
   end
   if invalid then
      print('invalid word <' .. word .. '> converted to <' .. newword .. '>')
   end
   return newword
end

local out = io.open(opt.arparep, 'w')
for line in io.lines(opt.arpa) do
   if line:match('\\data\\') then
      header = true
      ngram = false
      out:write(line)
      out:write('\n')
      goto continue
   end

   if line:match('\\end\\') then
      out:write(line)
      out:write('\n')
      goto exit
   end

   local n = line:match('^%s*\\(%d)%-grams%:%s*$')
   n = tonumber(n)
   if n then
      -- if n == 2 then
      --    goto exit
      -- end
      print('now dealing with ngrams with n=' .. n)
      ngram = n
      header = false
      format = "^%s*([e%-%.%d]+)"
      for i=1,n do
         format = format .. "%s+(%S+)"
      end
      format = format .. "(.*)$"
      out:write(line)
      out:write('\n')
      goto continue
   end

   if line:match('^%s*$') then
      out:write(line)
      out:write('\n')
      goto continue
   end

   if header then
      local n, count = line:match('^%s*ngram%s(%d)%=%s*(%d+)%s*')
      n = tonumber(n)
      count = tonumber(count)
      if n and count then
         print('n', n, 'count', count)
         out:write(line)
         out:write('\n')
         goto continue
      end
   end

   if ngram then
      local stuff = {line:match(format)}
      if stuff[1] then
         local score = table.remove(stuff, 1)
         local backoff = table.remove(stuff, ngram+1)
         backoff = backoff:match('(%S+)')
         for i=1,ngram do
            stuff[i] = checkword(stuff[i])
         end
         out:write(
            string.format(
               "%s\t%s%s%s\n",
               score,
               table.concat(stuff, " "),
               backoff and "\t" or "",
               backoff and backoff or ""
            )
         )
         goto continue
      end
   end

   error('cannot parse line <' .. line .. '>')
::continue::

end

::exit::

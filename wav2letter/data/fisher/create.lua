-- Fisher consists of the LDC datasets
-- LDC2005T19 - Fisher English Training Part 2, Transcripts
-- LDC2005S13 - Fisher English Training Part 2, Speech
-- LDC2004T19 - Fisher English Training Speech Part 1 Transcripts
-- LDC2004S13 - Fisher English Training Speech Part 1 Speech
-- There is no explicit evaluation or test data yet
-- I am assuming, that Part 2 and Part 1 transcripts follow the same guidelines!

local tnt = require 'fbtorchnet'
local paths = require 'paths'
local sndfile = require 'sndfile'
local xlua = require 'xlua'

require 'lfs'

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

local transPaths = {
   "Fisher_Part1",
   "Fisher_Part2",
}

local dataPaths = {
   "fe_03_p2_sph1/",
   "fe_03_p2_sph2/",
   "fe_03_p2_sph3/",
   "fe_03_p2_sph4/",
   "fe_03_p2_sph5/",
   "fe_03_p2_sph6/",
   "fe_03_p2_sph7/",
   "fisher_eng_tr_sp_d1",
   "fisher_eng_tr_sp_d2",
   "fisher_eng_tr_sp_d3",
   "fisher_eng_tr_sp_d4",
   "fisher_eng_tr_sp_d5",
   "fisher_eng_tr_sp_d6",
   "fisher_eng_tr_sp_d7",
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

local function createidx(src, dst)
   local trans = {}
   for _, p in pairs(transPaths) do
      fileapply(
         src .. '/' .. p,
         'fe_03_[%d]+.txt',
         function(filename)
            table.insert(trans, filename)
         end
      )
   end
   print(string.format("| Found %d transcriptions", #trans))

   local data = {}
   local datac = 0
   for _, p in pairs(dataPaths) do
      fileapply(
         src .. '/' .. p,
         '.wav',
         function(filename)
            local uttid = paths.basename(filename):sub(1, 11)
            data[uttid] = filename
            datac = datac + 1
         end
      )
   end
   print(string.format("| Found %d data files", datac))

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

   local uttIDidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/uttID.idx", dst),
      datafilename = string.format("%s/uttID.bin", dst),
      type = "byte"
   }

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

         --Open and read audio file
         local f = sndfile.SndFile(wav)
         local track = f:readFloat(f:info().frames)
         local info = f:info()

         local valid = true
         --Check if transcriptions don't try to break file boundaries
         for line in io.lines(filename) do
            if not line:match('#') and line:match('%S') then
               local s, e, _, lbl = line:match('^(%S+)%s+(%S+)%s+(%S):%s+(.*)$')
               s = tonumber(s)
               e = tonumber(e)
               assert(lbl)
               local start = s * info.samplerate
               local size = (e - s) * info.samplerate
               if start + size > track:size(1) then
                  valid = false
               end
            end
         end
         if not valid then
            print('Invalid transcription: ' .. filename)
         end
         while valid do
            local line = transText:read()
            if line  == nil then break end
            if not line:match('#') and line:match('%S') then
               --Getting start, end, channel and text
               local s, e, c, lbl = line:match('^(%S+)%s+(%S+)%s+(%S):%s+(.*)$')
               s = tonumber(s)
               e = tonumber(e)
               assert(lbl)

               local channel
               if c == 'A' then
                  channel = 1
               else
                  channel = 2
                  assert(c == 'B')
               end

               --Processing label
               lbl = lbl:lower()
               -- Removing noise annotations
               lbl = lbl:gsub('%[%S+%]', '')
               --Removing parentheses indicating insecurity of transcription (TODO: We could choose to exclude these examples)
               lbl = lbl:gsub('%(', '')
               lbl = lbl:gsub('%)', '')
               --There isn't much documentation on the meaning of '.', '_', ',', '?' and - but looking
               --at example files, there doesn't appear to be anything special about them
               --TODO: Investigate this more carefully! (Doesn't appear to be more info here https://catalog.ldc.upenn.edu/docs/LDC2004T19/)
               lbl = lbl:gsub('%.', ' ')
               lbl = lbl:gsub('_', ' ')
               lbl = lbl:gsub('-', '')
               lbl = lbl:gsub(',', '')
               lbl = lbl:gsub('?', '')
               lbl = lbl:gsub('%s+', ' ')
               lbl = lbl:gsub('^%s+', '')
               lbl = lbl:gsub('%s+$', '')

               local start = s * info.samplerate
               local size = (e - s) * info.samplerate

               local sane = true
               --Very few utterances (<50) contain an ampersand, but it's not always clear how it should be pronounced
               if string.find(lbl, '&') then
                  sane = false
               end
               --The one transcription example that I found that contains this had a very heavily accented bad speaker
               --The fisher readme doesn't appear to mention this symbol, so I'll ignore utterances (count of 2) that use this
               if string.find(lbl, '*') then
                  sane = false
               end
               --This symbols doesn't really seem to mean anything, but I'll just the one utterance that uses it.
               if string.find(lbl, '~') then
                  sane = false
               end

               --Sometimes it's not clear how the numbers are pronounced (e.g. 401k four - o - one k)
               --This excludes another 82 utterances
               if string.find(lbl, '[0-9]') then
                  sane = false
               end

               --Only 3 utterances contain this (and one of them is in german)
               if string.find(lbl, '<') or string.find(lbl, '>') then
                  sane = false
               end

               if sane and lbl:match('[^a-z%s\']') then
                  error("Improperly processed " .. uttid)
               end

               --Ignore anything shorter than 2 seconds
               if size / info.samplerate > 2 and sane and lbl ~= "" then

                  --Creating temporary file in memory
                  local tmpfname = '/dev/shm/838145961926764.wav'
                  local out = sndfile.SndFile(tmpfname, 'w', {samplerate=info.samplerate, channels=1, format="WAV", subformat="ULAW"})
                  local input = track:narrow(1, start + 1, size):narrow(2, channel, 1)
                  allSeconds = allSeconds + (size / info.samplerate)
                  uttCount = uttCount + 1
                  out:writeFloat(input)
                  out:close()

                  --Store bytes representing file
                  local fin = torch.DiskFile(tmpfname)
                  fin:seekEnd()
                  local size = fin:position()
                  fin:seek(1)
                  local stor = fin:readByte(size-1)
                  fin:close()

                  inputidx:add(torch.ByteTensor(stor))

                  targetidx:add(torch.ByteTensor(torch.ByteStorage():string(lbl)))

                  --Store utterance id + original time stamps
                  uttIDidx:add(torch.ByteTensor(torch.ByteStorage():string(uttid .. '-' .. s .. '-' .. e)))

               end
            end
         end
      end
      progress()
   end

   inputidx:close()
   targetidx:close()
   uttIDidx:close()

   print(string.format("Written %d hours of speech", allSeconds / (60 * 60)))
   print(string.format("Written %d utterances", uttCount))
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

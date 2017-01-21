-- librispeech
-- targets are the transcription strings viewed as ByteTensors

local tnt = require 'torchnet'

require 'lfs'

local function parse_speakers_gender(lines)
   local ret = {}
   for id, g in lines:gmatch("(%d+)[ |]+(%u)[^\n]+[\n]*") do
      ret[tonumber(id)] = g == 'M' and 1 or 0
   end
   return ret
end
print('| parsing SPEAKERS.TXT for gender...')
local gender = parse_speakers_gender(io.open(string.format('%s/SPEAKERS.TXT',
                                             arg[1]), 'r'):read("*all"))

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

local function createidx(src, dst)
   local trans = {}
   assert(type(src) == 'string', 'src path should be a string')
   print(string.format('| analyzing %s...', src))
   fileapply(
      src,
      '%.trans%.txt$',
      function(filename)
         table.insert(trans, filename)
      end
   )
   table.sort(
      trans,
      function(a, b)
         return a < b
      end
   )
   print(string.format('| writing %s...', dst))

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

   local speakeridx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/speaker.idx", dst),
      datafilename = string.format("%s/speaker.bin", dst),
      type = "int"
   }

   local filenameidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/filename.idx", dst),
      datafilename = string.format("%s/filename.bin", dst),
      type = "byte"
   }

   for _, filename in pairs(trans) do
      for line in io.lines(filename) do
         local input, lbl = line:match('^(%S+)%s+(.*)$')
         assert(input and lbl)

         inputidx:add(string.format('%s%s.flac', filename:gsub('[^/]+$', ''), input))

         lbl = lbl:gsub('^%s+', '')
         lbl = lbl:gsub('%s+$', '')
         lbl = lbl:lower()
         targetidx:add(torch.ByteTensor(torch.ByteStorage():string(lbl)))
         local spkr = filename:match('/(%d+)[^/]+$')
         spkr = tonumber(spkr)
         speakeridx:add(torch.IntTensor({spkr, gender[spkr]}))
         filenameidx:add(torch.ByteTensor(torch.ByteStorage():string(input)))
      end
   end

   inputidx:close()
   targetidx:close()
   filenameidx:close()
   speakeridx:close()
end

assert(#arg == 2, string.format('usage: %s <src dir> <dst dir>', arg[0]))
local src = arg[1]
local dst = arg[2]

local subpaths = {
   'train-clean-100', 'train-clean-360', 'train-other-500',
   'dev-clean', 'dev-other',
   'test-clean', 'test-other'
}

-- create indexed datasets
for _, subpath in ipairs(subpaths) do
   local src = string.format("%s/%s", src, subpath)
   local dst = string.format("%s/%s", dst, subpath)
   os.execute(string.format('mkdir -p %s', dst))
   createidx(src, dst)
end

-- create letters list
print('| creating letter list...')
local alltargets = {}
for _, subpath in ipairs(subpaths) do
   local path = string.format("%s/%s", dst, subpath)
   table.insert(
      alltargets,
      tnt.IndexedDataset{
         path = path,
         fields = {"target"}
      }
   )
end
alltargets = tnt.ConcatDataset{
   datasets = alltargets
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

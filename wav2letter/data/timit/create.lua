-- timit phones
-- core test set setup

local tnt = require 'fbtorchnet'
local paths = require 'paths'

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


local function createidx(src, dst, listname, phones)
   print(string.format('| writing %s...', dst))
   os.execute(string.format('mkdir -p %s', dst))

   local inputidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/input.idx", dst),
      datafilename = string.format("%s/input.bin", dst),
      type = "byte"
   }

   local targetidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/target.idx", dst),
      datafilename = string.format("%s/target.bin", dst),
      type = "long"
   }

   local filenameidx = tnt.IndexedDatasetWriter{
      indexfilename = string.format("%s/filename.idx", dst),
      datafilename = string.format("%s/filename.bin", dst),
      type = "byte"
   }

   for filename in io.lines(listname) do
      local target = {}
      local targetfilename = filename:gsub('%.wav$', '%.phn')
      for line in io.lines(src .. "/timit/" .. targetfilename) do
         local s, e, p = line:match('(%S+)%s+(%S+)%s+(%S+)')
         p = phones[p]
         assert(p)
         table.insert(target, p)
      end
      target = torch.LongTensor(target)

      inputidx:add(src .. "/timit/" .. filename)
      targetidx:add(target)
      filenameidx:add(torch.ByteTensor(torch.ByteStorage():string(filename)))
   end

   inputidx:close()
   targetidx:close()
   filenameidx:close()
end

assert(#arg == 2, string.format('usage: %s <src dir> <dst dir>', arg[0]))
local src = arg[1]
local dst = arg[2]

local phones = {}
print(string.format('| analyzing %s...', src .. "/timit/train"))
fileapply(
   src .. "/timit/train",
      '%.phn',
      function(filename)
         for line in io.lines(filename) do
            local s, e, p = line:match('(%S+)%s+(%S+)%s+(%S+)')
            assert(s and e and p)
            if not phones[p] then
               table.insert(phones, p)
               phones[p] = #phones
            end
         end
      end
)
table.sort(
   phones,
   function(a, b)
      return a < b
   end
)
print(string.format('| %d phones found', #phones))
os.execute(string.format('mkdir -p %s', dst))
local f = io.open(dst .. "/phones.lst", "w")
for idx, phone in ipairs(phones) do
   phones[phone] = idx
   f:write(phone .. '\n')
end
f:close()
for idx, phone in ipairs(phones) do
   assert(phones[phone] == idx)
end

createidx(src, dst .. "/train/", paths.thisfile("train.lst"), phones)
createidx(src, dst .. "/valid/", paths.thisfile("valid.lst"), phones)
createidx(src, dst .. "/test/",  paths.thisfile("test.lst"),  phones)

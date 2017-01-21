local tnt = require 'fbtorchnet'
local sndfile = require 'sndfile'
local utils = require 'wav2letter.utils'

local function as(opt, field, typename)
   assert(opt[field] ~= nil, 'option ' .. field .. ' not set.')
   assert(type(opt[field]) == typename , 'option ' .. field .. ' of wrong type.')
   return opt[field]
end


local function config(opt)
   local datadir      = as(opt, "datadir"     , "string"  )
   local numwords     = as(opt, "words"       , "number"  )
   local lsc100       = as(opt, "lsc100"      , "boolean" )
   local lsc360       = as(opt, "lsc360"      , "boolean" )
   local lso500       = as(opt, "lso500"      , "boolean" )
   local lmessenger   = as(opt, "lmessenger"  , "boolean" )
   local lfisher      = as(opt, "lfisher"     , "boolean" )
   local lswb         = as(opt, "lswb"     , "boolean" )
   local maxloadseed  = as(opt, "maxloadseed" , "number"  )
   local maxload      = as(opt, "maxload"     , "number"  )
   local maxloadvalid = as(opt, "maxloadvalid", "number"  )
   local maxloadtest  = as(opt, "maxloadtest" , "number"  )
   local mfsc         = as(opt, "mfsc"        , "boolean" )
   local pow          = as(opt, "pow"         , "boolean" )
   local mfcc         = as(opt, "mfcc"        , "boolean" )
   local mfcc_coefs   = as(opt, "mfcc_coefs"  , "number"  )
   local ctc          = as(opt, "ctc"         , "boolean" )
   local surround     = as(opt, "surround"    , "boolean" )
   local l8khz        = as(opt, "l8khz"       , "boolean" )
   local config = {datasets = {}}
   local roots = {}
   roots.fisher      = string.format('%s/fisher-idx',      datadir)
   roots.librispeech = string.format('%s/librispeech-idx', datadir)
   roots.messenger   = string.format('%s/messenger-idx',   datadir)
   roots.swb         = string.format('%s/swb-idx',         datadir)
--   roots.wsj         = string.format('%s/wsj-idx',         datadir)

   local words = {}
   if numwords > 0 then
      local function addtodict(word)
         assert(type(word) == 'string')
         assert(not words[word], 'duplicates in provided dictionary')
         local idx = #words + 1
         words[idx] = word
         words[word] = idx
      end
      -- get words.lst with the count-words.lua util in data/
      for word in io.lines(string.format('%s/words.lst', root)) do
         word = word:match('^(%S+)')
         addtodict(word)
         if #words == numwords then
            break
         end
      end
      print(string.format('| considering a vocabulary of %d words (+SIL +UNK)', #words))
      addtodict('SIL') -- sil
      addtodict('UNK') -- unknown
   end

   local function tblequal(t1, t2)
      local equal = true
      for k, v in pairs(t1) do equal = equal and (t2[k] ~= nil) end
      for k, v in pairs(t2) do equal = equal and (t1[k] ~= nil) end
      if equal then
         for k, v in pairs(t2) do
            equal = equal and (t1[k] == t2[k])
         end
      end
      return equal
   end

   local function buildc2l(root)
      local class2letter = {}
      for letter in io.lines(string.format('%s/letters.lst', root)) do
         table.insert(class2letter, string.byte(letter))
      end
      return class2letter
   end

   local class2letter
   for _, root in pairs(roots) do
      if not class2letter then
         class2letter = buildc2l(root)
      else
         assert(tblequal(class2letter, buildc2l(root)), string.format(root .. ' incompatible letter.lst'))
      end
   end

   class2letter = torch.LongTensor(class2letter)
   local letter2class = torch.LongTensor(class2letter:max()):zero()
   for idx=1,class2letter:size(1) do
      letter2class[class2letter[idx]] = idx
   end

   function config.class2letter(output)
      return class2letter:gather(1, output:long())
   end

   local function audioReader(ds)
      local tmpTensor = torch.FloatTensor()
      return function (input)
         local f = sndfile.SndFile(input:storage())
         local info = f:info()
         local sr = info.samplerate
         assert(sr == 8000 or sr == 16000, 'invalid samplerate')
         input = f:readFloat(info.frames)
         f:close()
         if ds and sr == 16000 then
            local i = input:unfold(1, 1, 2)
            tmpTensor:resizeAs(i):copy(i)
            return tmpTensor:view(i:size(1), 1)
         else
            return input
         end
      end
   end

   local function makeReader(path, maxload)
      return tnt.TransformDataset{
         dataset = tnt.IndexedDataset{
            path = path,
            fields = {"input", "target"},
            maxload = maxload,
         },
         transform = audioReader(l8khz),
         key = 'input'
      }
   end

   local tp = {}
   if lsc100     then table.insert(tp, string.format('%s/train-clean-100', roots.librispeech)) end
   if lsc360     then table.insert(tp, string.format('%s/train-clean-360', roots.librispeech)) end
   if lso500     then table.insert(tp, string.format('%s/train-other-500', roots.librispeech)) end
   if lfisher    then table.insert(tp, roots.fisher) end
   if lswb       then table.insert(tp, roots.swb) end
   if lmessenger then table.insert(tp, roots.messenger) end

   assert(#tp > 0, 'Need to select at least some training data!')

   if lfisher or lswb then
      assert(l8khz, 'swb and fisher use 8khz, so we need to downsample other sources!')
   end

   config.traindataset =
      function()
         local traindatasets = {}
         for _, path in ipairs(tp) do
            table.insert(traindatasets, makeReader(path))
         end
         local all = tnt.ConcatDataset{
            datasets = traindatasets
         }
         local gen = torch.Generator()
         torch.manualSeed(gen, maxloadseed)
         local subset = torch.randperm(gen, all:size())
         return tnt.ResampleDataset{
            dataset = all,
            size = maxload,
            sampler = function(dataset, idx) return subset[idx] end
         }
      end

   config.validdatasets = {
      ["dev-clean"] = function() return makeReader(string.format('%s/dev-clean', roots.librispeech), maxloadvalid) end,
      ["dev-other"] = function() return makeReader(string.format('%s/dev-other', roots.librispeech), maxloadvalid) end
   }

   config.testdatasets = {
      ["test-clean"] = function() return makeReader(string.format('%s/test-clean', roots.librispeech), maxloadtest) end,
      ["test-other"] = function() return makeReader(string.format('%s/test-other', roots.librispeech), maxloadtest) end
   }

   -- basic specs
   config.specs = {
      nchannel = (mfsc and 40 ) or ((pow and 257 ) or (mfcc and mfcc_coefs*3 or 1)),
      samplerate = (l8khz and 8000) or 16000,
      nclass = numwords > 0 and #words or class2letter:size(1),
   }
   --blank label for ctc
   config.specs.nclass = config.specs.nclass + ((ctc and 1) or 0)

   -- transforms
   local transforms = {}

   if numwords > 0 then
      transforms.target =
         function(target)
            target = target:clone():storage():string()
            local newtarget = {words.SIL}
            for word in target:gmatch('(%S+)') do
               table.insert(newtarget, words[word] or words.UNK)
               table.insert(newtarget, words.SIL)
            end
            newtarget = torch.LongTensor(newtarget)
            return newtarget
         end

      config.tostring =
         function(output)
            local str = {}
            assert(output:nDimension() == 1)
            for i=1,output:size(1) do
               local word = words[output[i]]
               assert(word)
               table.insert(str, word)
            end
            return table.concat(str)
         end
   else
      -- target
      transforms.target =
         function(target)
            if surround then
               --surround target with spaces
               local tp2 = target:clone():resize(target:size(1) + 2)
               tp2[1] = string.byte(' ')
               tp2[target:size(1) + 2] = string.byte(' ')
               tp2:narrow(1, 2, target:size(1)):copy(target)
               target = tp2
            end
            return letter2class:gather(1, target:long())
         end

      -- print remap'ed output
      config.tostring =
         function(output)
            return config.class2letter(output):byte():storage():string()
         end
   end

   config.transforms = transforms

   return config
end

return config

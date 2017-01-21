local tnt = require 'fbtorchnet'
local sndfile = require 'sndfile'

local function config(opt)
   local config = {datasets = {}}
   local root = string.format('%s/wsj-idx', opt.datadir)

   local class2letter = {}
   for letter in io.lines(string.format('%s/letters.lst', root)) do
      table.insert(class2letter, string.byte(letter))
   end
   -- for i=1,opt.replabel do -- FIX: so tostring() works
   --    table.insert(class2letter, string.byte('' .. opt.replabel - i + 1)) -- FIX: order
   -- end
   class2letter = torch.LongTensor(class2letter)
   local letter2class = torch.LongTensor(class2letter:max()):zero()
   for idx=1,class2letter:size(1) do
      letter2class[class2letter[idx]] = idx
   end

   local function dataset(subpath, maxload)
      return function()
         local dataset = tnt.TransformDataset{
            dataset = tnt.IndexedDataset{
               path = paths.concat(root, subpath),
               fields = {"data"},
               maxload = maxload,
            },
            transform =
               function(sample)
                  sample = sample.data
                  local f = sndfile.SndFile(sample.wav)
                  local info = f:info()
                  assert(info.samplerate == config.specs.samplerate,
                         'invalid samplerate')
                  local input = f:readFloat(info.frames)
                  f:close()
                  local target = torch.ByteTensor(
                     torch.ByteStorage():string(sample.spellings))
                  if opt.surround then
                     --surround target with spaces
                    if target:nDimension() == 0 then
                       target = target.new(1):fill(string.byte(' '))
                    else
                        local tp2 = target:clone():resize(target:size(1) + 2)
                        tp2[1] = string.byte(' ')
                        tp2[target:size(1) + 2] = string.byte(' ')
                        tp2:narrow(1, 2, target:size(1)):copy(target)
                        target = tp2
                    end
                  end
                  target = letter2class:gather(1, target:long())
                  return {input=input, target=target, words=sample.words}
               end
         }
         return dataset
      end
   end

   if opt.si84 then -- rarely used
      config.traindataset = dataset("si84", opt.maxload)
   else
      config.traindataset = dataset("si284", opt.maxload)
   end

   config.validdatasets = {
      nov93dev = dataset("nov93dev", opt.maxloadvalid)
   }

   config.testdatasets = {
      nov92 = dataset("nov92", opt.maxloadtest),
      nov93 = dataset("nov93", opt.maxloadtest)
   }

   -- basic specs
   config.specs = {
      nchannel =
         (opt.mfsc and 40 )
         or ((opt.pow and 257 )
         or (opt.mfcc and opt.mfcc_coefs*3 or 1)),
      samplerate = 16000,
      nclass = class2letter:size(1),
   }
   --blank label for ctc
   config.specs.nclass = config.specs.nclass + ((opt.ctc and 1) or 0)

   config.tostring =
      function(output)
         print(output:long():min(), output:long():max(), class2letter:size(1))
         return class2letter:gather(1, output:long()):byte():storage():string()
      end

   config.transforms = {}

   return config
end

return config

local tnt = require 'torchnet'
local sndfile = require 'sndfile'

-- should i really discard q?
local dict39 = {
   ao = "aa",
   ax = "ah",
   ["ax-h"] = "ah",
   axr = "er",
   hv = "hh",
   ix = "ih",
   el = "l",
   em = "m",
   en = "n",
   nx = "n",
   eng = "ng",
   zh = "sh",
   ux = "uw",
   pcl = "h#",
   tcl = "h#",
   kcl = "h#",
   bcl = "h#",
   dcl = "h#",
   gcl = "h#",
   pau = "h#",
   ["#h"] = "h#",
   epi = "h#"
}

local function config(opt)
   local config = {datasets = {}}
   local root = string.format('%s/timit-idx', opt.datadir)
   if opt.discardq then
      dict39["q"] = "h#"
   end

   local phones61 = {}
   for phone61 in io.lines(string.format('%s/phones.lst', root)) do
      table.insert(phones61, phone61)
      phones61[phone61] = #phones61
   end

   local phones39 = {}
   for _, phone61 in ipairs(phones61) do
      local phone39 = dict39[phone61] or phone61
      if not phones39[phone39] then
         table.insert(phones39, phone39)
         phones39[phone39] = #phones39
      end
   end
   table.sort(
      phones39,
      function(a, b)
         return a < b
      end
   )
   for idx39, phone39 in ipairs(phones39) do
      phones39[phone39] = idx39
   end

   local phones61To39 = torch.LongTensor(#phones61):zero()
   for idx61, phone61 in ipairs(phones61) do
      local phone39 = dict39[phone61] or phone61
      local idx39 = phones39[phone39]
      assert(idx39)
      phones61To39[idx61] = idx39
   end

   config.traindataset =
      function()
         return tnt.ShuffleDataset{
            dataset = tnt.IndexedDataset{
               path = string.format('%s/train', root),
               fields = {"input", "target"},
               maxload = opt.maxload
            }
         }
      end

   config.validdatasets = {
      valid =
         function()
            return tnt.IndexedDataset{
               path = string.format('%s/valid', root),
               fields = {"input", "target"},
               maxload = opt.maxloadvalid
            }
         end
   }

   config.testdatasets = {
      test =
         function()
            return tnt.IndexedDataset{
               path = string.format('%s/test', root),
               fields = {"input", "target"},
               maxload = opt.maxloadtest
            }
         end
   }

   -- basic specs
   config.specs = {
      nchannel = opt.mfcc and opt.mfcc_coefs*3 or 1,
      samplerate = 16000,
      nclass = opt.dict39 and #phones39 or #phones61,
   }
   --blank label for ctc
   config.specs.nclass = config.specs.nclass + ((opt.ctc and 1) or 0)

   -- transforms
   local transforms = {}

   -- input
   transforms.input =
      function(input)
         local f = sndfile.SndFile(input:storage())
         local info = f:info()
         assert(info.samplerate == config.specs.samplerate,
                                               'invalid samplerate')
         input = f:readFloat(info.frames)
         f:close()
         return input
      end

   if opt.dict39 then
      -- target [transform target into 39-phones based]
      transforms.target =
         function(target)
            return phones61To39:gather(1, target:long())
         end
   else
      -- remap [do not transform target during training,
      -- but convert output/target into 39-phones-based for eval]
      transforms.remap =
         function(output)
            return phones61To39:gather(1, output:long())
         end
   end

   config.transforms = transforms

   -- print remap'ed output
   config.tostring =
      function(target)
         local txt = {}
         target:apply(
            function(c)
               local p = phones39[c]
               assert(p, 'invalid phone index')
               table.insert(txt, p)
            end
         )
         return table.concat(txt, ' ')
      end

   return config
end

return config

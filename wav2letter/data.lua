local argcheck = require 'argcheck'
local tnt = require 'torchnet'
require 'torchnet.sequential'
local threads = require 'threads'
local readers = require 'wav2letter.readers'
require 'wav2letter' -- for numberedfilesdataset
local data = {}
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
   epi = "h#",
   q = "h#"
}

local function batchmerge(sample)
   local pad = 0
   local imax = 0
   local channels
   for _,input in ipairs(sample.input) do
      imax = math.max(imax, input:size(1))
      channels = channels or input:size(2)
   end
   local mergeinput = torch.Tensor(#sample.input, imax, channels):fill(0)
   for i,input in ipairs(sample.input) do
      mergeinput[i]:narrow(1, 1, input:size(1)):copy(input)
   end
   return {input=mergeinput, target=sample.target}
end

data.dictcollapsephones = argcheck{
   noordered = true,
   {name='dictionary', type='table'},
   call =
      function(dict)
         local cdict = {}
         for _, phone in ipairs(dict) do
            if not dict39[phone] then
               data.dictadd{dictionary=cdict, token=phone}
            end
         end
         for _, phone in ipairs(dict) do
            if dict39[phone] then
               data.dictadd{dictionary=cdict, token=phone, idx=assert(cdict[dict39[phone]])}
            end
         end
         return cdict
      end
}

data.dictadd = argcheck{
   noordered = true,
   {name='dictionary', type='table'},
   {name='token', type='string'},
   {name='idx', type='number', opt=true},
   call =
      function(dict, token, idx)
         local idx = idx or #dict+1
         assert(not dict[token], 'duplicate entry name in dictionary')
         dict[token] = idx
         if not dict[idx] then
            dict[idx] = token
         end
      end
}

data.newdict = argcheck{
   {name='path', type='string'},
   call =
      function(path)
         local dict = {}
         for line in io.lines(path) do
            local token, idx = line:match('^(%S+)%s*(%d+)$')
            idx = tonumber(idx)
            if token and idx then
               data.dictadd{dictionary=dict, token=token, idx=idx}
            else
               data.dictadd{dictionary=dict, token=line}
            end
         end
         return dict
      end
}

data.dictmaxvalue =
   function(dict)
      local maxvalue = 0
      for k, v in pairs(dict) do
         maxvalue = math.max(maxvalue, v)
      end
      return maxvalue
   end

data.newsampler =
   function()
      local resampleperm = torch.LongTensor()
      local function resample()
         resampleperm:resize(0)
      end
      local sampler =
         threads.safe(
            function(dataset, idx)
               if resampleperm:nDimension() == 0 then
                  print(string.format('# resampling: size=%d', dataset:size()))
                  resampleperm:randperm(dataset:size())
               end
               return resampleperm[idx]
            end
         )
      return sampler, resample
   end

data.namelist = argcheck{
   {name='names', type='string'},
   call =
      function(names)
         local list = {}
         for name in names:gmatch('(%S+)') do
            table.insert(list, name)
         end
         return list
      end
}

data.newdataset = argcheck{
   noordered = true,
   {name='path', type='string'},
   {name='features', type='table'}, -- {name= [alias=] reader=}
   {name='names', type='table', opt=true}, -- concat?
   {name='transforms', type='table', opt=true}, -- <alias>: function
   {name='maxload', type='number', opt=true},
   {name='batch', type='number', opt=true},
   {name='batchresolution', type='number', opt=true},
   {name='sampler', type='function', opt=true},
   call =
      function(path, features, names, transforms, maxload, batch, batchresolution, sampler)
         if names then
            assert(#names > 0, 'names should be a list of strings')
            if #names == 1 then
               path = paths.concat(path, names[1])
               names = nil
            end
         end

         local dataset
         if names then -- concat several datasets?
            local datasets = {}
            for _, name in ipairs(names) do
               table.insert(
                  datasets,
                  data.newdataset{
                     path = paths.concat(path, name),
                     features = features,
                     transforms = transforms,
                  }
               )
            end
            dataset = tnt.ConcatDataset{datasets = datasets}

            -- brutal cut (one could allow pre-shuffling)
            if maxload and maxload > 0 then
               dataset = tnt.ResampleDataset{
                  dataset = dataset,
                  size = maxload
               }
            end
         else -- read one dataset
            dataset = tnt.NumberedFilesDataset{
               path = path,
               features = features,
               names = names,
               maxload = maxload
            }
         end

         -- transform it?
         if transforms then
            dataset = tnt.TransformDataset{
               dataset = dataset,
               transforms = transforms
            }
         end

         -- batch it?
         if batch and batch > 0 then
            local iszdataset = data.newdataset{
               path = path,
               features = {{name="isz", reader=readers.number{}}},
               maxload = maxload
            }
            dataset = tnt.BatchDataset{
               dataset = tnt.BucketSortedDataset{
                  dataset = dataset,
                  resolution = batchresolution,
                  samplesize =
                     function(dataset, idx)
                        return iszdataset:get(idx).isz
                     end
               },
               merge = batchmerge,
               batchsize = batch,
            }
         end

         -- resample it?
         if sampler then
            dataset = tnt.ResampleDataset{
               dataset = dataset,
               sampler = sampler
            }
         end

         return dataset
      end
}

data.newfilterbysize = argcheck{
   noordered = true,
   {name='kw', type='number'},
   {name='dw', type='number'},
   {name='minisz', type='number', default=0},
   {name='maxisz', type='number', default=math.huge},
   {name='mintsz', type='number', default=0},
   {name='maxtsz', type='number', default=math.huge},
   {name='batchsize', type='number', default=0},
   {name='shift', type='number', default=0},
   call =
      function(kw, dw, minisz, maxisz, mintsz, maxtsz, batchsize, shift)
         local function check(input, target)
            local isz = input:size(1)
            local tsz = target:size(1)
            if isz < math.max(kw+tsz*dw, minisz) or isz > maxisz then
               print("I")
               return false
            end
            if tsz < mintsz or tsz > maxtsz then
               print("T")
               return false
            end
            return true
         end

         return function(sample)
            -- with opt.shift last one is smaller
            local input = shift > 0 and sample.input[#sample.input] or sample.input
            local target = sample.target
            if batchsize > 0 then
               for i=1,input:size(1) do
                  if not check(input[i], target[i]) then
                     return false
                  end
               end
               return true
            else
               return check(input, target)
            end
         end
      end
}

return data

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
                  print(string.format('| resampling: size=%d', dataset:size()))
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
         for name in names:gmatch('([^%+]+)') do
            table.insert(list, name)
         end
         return list
      end
}

data.label2string = argcheck{
   {name='labels', type='torch.LongTensor'},
   {name='dict', type='table'},
   {name='spacing', type='string', default=''},
   call =
      function(tensor, dict, spc)
         local str = {}
         assert(tensor:nDimension() == 1, '1d tensor expected')
         for i=1,tensor:size(1) do
            local lbl = dict[tensor[i]]
            if not lbl then
               error(string.format("unknown label <%s>", tensor[i]))
            end
            table.insert(str, lbl)
         end
         return table.concat(str, spc)
      end
}

data.transform = argcheck{
   {name='dataset', type='tnt.Dataset'},
   {name='transforms', type='table', opt=true},
   call =
      function(dataset, transforms)
         if transforms then
            return tnt.TransformDataset{
               dataset = dataset,
               transforms = transforms
            }
         else
            return dataset
         end
      end
}

data.partition = argcheck{
   {name='dataset', type='tnt.Dataset'},
   {name='n', type='number'},
   {name='id', type='number'},
   call =
      function(dataset, n, id)
         assert(id >= 1 and id <= n, "invalid id range")
         if n == 1 then
            return dataset
         else
            local partitions = {}
            for i=1,n do
               partitions[tostring(i)] = math.floor(dataset:size()/n)
            end
            return tnt.SplitDataset{
               dataset = dataset,
               partitions = partitions,
               initialpartition = "" .. id
            }
         end
      end
}

data.resample = argcheck{
   {name='dataset', type='tnt.Dataset'},
   {name='sampler', type='function', opt=true},
   {name='size', type='number', opt=true},
   call =
      function(dataset, sampler, size)
         if sampler then
            return tnt.ResampleDataset{
               dataset = dataset,
               sampler = sampler,
               size = size
            }
         else
            return dataset
         end
      end
}

data.filtersizesampler = argcheck{
   {name='sizedataset', type='tnt.Dataset'},
   {name='filter', type='function'},
   call =
      function(sizedataset, filter)
         local perm = torch.zeros(sizedataset:size())
         local size = 0
         for i=1,sizedataset:size() do
            local sz = sizedataset:get(i)
            assert(sz.isz and sz.tsz, 'sizedataset:get() should return {isz=, tsz=}')
            if filter(sz.isz, sz.tsz) then
               size = size + 1
               perm[size] = i
            end
         end
         print(string.format("| %d/%d filtered samples", size, sizedataset:size()))
         return
            function(_, idx)
               return perm[idx]
            end, size
      end
}

data.mapconcat = argcheck{
   {name='closure', type='function'},
   {name='args', type='table'},
   {name='maxload', type='number', opt=triue},
   call =
      function(closure, args, maxload)
         local datasets = {}
         for i, arg in ipairs(args) do
            datasets[i] = closure(arg)
         end
         local dataset = tnt.ConcatDataset{datasets = datasets}
         -- brutal cut (one could allow pre-shuffling)
         if maxload and maxload > 0 then
            dataset = tnt.ResampleDataset{
               dataset = dataset,
               size = maxload
            }
         end
         return dataset
      end
}

data.batch = argcheck{
   {name='dataset', type='tnt.Dataset'},
   {name='sizedataset', type='tnt.Dataset'},
   {name='batchsize', type='number'},
   {name='batchresolution', type='number'},
   call =
      function(dataset, sizedataset, batchsize, batchresolution)
         assert(dataset:size() == sizedataset:size(), 'dataset and sizedataset do not have the same size')
         if batchsize <= 0 then
            return dataset
         else
            return tnt.BatchDataset{
               dataset = tnt.BucketSortedDataset{
                  dataset = dataset,
                  resolution = batchresolution,
                  samplesize =
                     function(dataset, idx)
                        local isz = sizedataset:get(idx).isz
                        assert(type(isz) == 'number', 'isz size feature nil or not a number')
                        return isz
                     end
               },
               batchsize = batchsize,
               merge = batchmerge,
            }
         end
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
         return function(isz, tsz)
            if isz < math.max(kw+tsz*dw, minisz) or isz > maxisz then
               return false
            end
            if tsz < mintsz or tsz > maxtsz then
               return false
            end
            return true
         end
      end
}

return data

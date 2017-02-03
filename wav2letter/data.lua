local argcheck = require 'argcheck'
local tnt = require 'torchnet'
local threads = require 'threads'
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
   {name='sampler', type='function', opt=true},
   call =
      function(path, features, names, transforms, maxload, batch, sampler)
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
            error("batch is NYI")
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

local function filterbysize(sample)
   -- with opt.shift last one is smaller
   local input = opt.shift > 0 and sample.input[#sample.input] or sample.input
   local target = sample.target
   local isz = opt.batchsize > 1 and input[1]:size(1) or input:size(1)
   local tsz = opt.batchsize > 1 and target[1]:size(1) or target:size(1)
   if isz < kw+tsz*dw then
      return false
   end
   if opt.batchsize > 1 then
      for i = 2, #target do
         local iszI = input[i]:size(1)
         local tszI = target[i]:size(1)
         if iszI < kw+tszI*dw then
            return false
         end
         tsz = math.max(tsz, tszI)
         isz = math.max(isz, iszI)
      end
   end
   if opt.maxisz > 0 and isz > opt.maxisz then
      return false
   end
   if tsz < opt.mintsz then
      print("warning ca va peter -- filtered out")
      return false
   end
   if opt.maxtsz > 0 and tsz > opt.maxtsz then
      return false
   end
   return true
end

return data

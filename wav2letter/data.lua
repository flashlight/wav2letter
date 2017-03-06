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
   local mergeinput = sample.input[1].new(#sample.input, imax, channels):fill(0)
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

data.newdataset = argcheck{
   noordered = true,
   {name="names", type="table"},
   {name="opt", type="table"},
   {name="dict", type="table"},
   {name="kw", type="number"},
   {name="dw", type="number"},
   {name="sampler", type="function", opt=true},
   {name="mpirank", type="number", default=1},
   {name="mpisize", type="number", default=1},
   {name="maxload", type="number", opt=true},
   {name="aug", type="boolean", opt=true},
   call =
      function(names, opt, dict, kw, dw, sampler, mpirank, mpisize, maxload, aug)
         local tnt = require 'torchnet'
         local data = paths.dofile('data.lua')
         local readers = require 'wav2letter.readers'
         local transforms = paths.dofile('transforms.lua')
         require 'wav2letter'
         local inputtransform, inputsizetransform = transforms.input{
            aug = opt.aug and {
               samplerate = opt.samplerate,
               bendingp = opt.augbendingp,
               speedp = opt.augspeedp,
               speed = opt.augspeed,
               chorusp = opt.chorusp,
               echop = opt.echop,
               compandp = opt.compandp,
               flangerp = opt.flangerp,
               noisep = opt.noisep,
               threadid = __threadid
                              } or nil,
            mfcc = opt.mfcc and {
               samplerate = opt.samplerate,
               coeffs = opt.mfcccoeffs,
               melfloor = opt.melfloor
                                } or nil,
            mfsc = opt.mfsc and {
               samplerate = opt.samplerate,
               melfloor = opt.melfloor
                                } or nil,
            pow = opt.pow and {
               samplerate = opt.samplerate,
               melfloor = opt.melfloor
                              } or nil,
            normmax = opt.inormmax,
            normloc = opt.inormloc and {kw=opt.inkw, dw=opt.indw, noisethreshold=opt.innt} or nil,
            norm = not opt.inormax and not opt.inormloc,
            pad = {size=kw/2, value=0},
         }
         local targettransform, targetsizetransform = transforms.target{
            surround = opt.surround ~= '' and assert(dict[opt.surround], 'invalid surround label') or nil,
            replabel = opt.replabel > 0 and {n=opt.replabel, dict=dict} or nil,
            uniq = true,
         }

         local function datasetwithfeatures(features, transforms)
            return data.transform{
               dataset = data.partition{
                  dataset = data.mapconcat{
                     closure = function(name)
                        return tnt.NumberedFilesDataset{
                           path = paths.concat(opt.datadir, name),
                           features = features,
                        }
                     end,
                     args = names,
                     maxload = maxload
                  },
                  n = mpisize,
                  id = mpirank
               },
               transforms = transforms
            }
         end

         local dataset = datasetwithfeatures(
            {
               {
                  name = opt.input,
                  alias = "input",
                  reader = readers.audio{
                     samplerate = opt.samplerate,
                     channels = opt.channels
                  },
               },
               {
                  name = opt.target,
                  alias = "target",
                  reader = readers.tokens{
                     dictionary = dict
                  }
               },
            },
            {
               input = inputtransform,
               target = targettransform
            }
         )

         local sizedataset = datasetwithfeatures(
            {
               {name = opt.input .. "sz", alias = "isz", reader = readers.number{}},
               {name = opt.target .. "sz", alias = "tsz", reader = readers.number{}}
            },
            {
               isz = inputsizetransform,
               tsz = targetsizetransform,
            }
         )

         -- filter
         local filter = data.newfilterbysize{
            kw = kw,
            dw = dw,
            minisz = opt.minisz,
            maxisz = opt.maxisz,
            maxtsz = opt.maxtsz,
            batchsize = opt.batchsize,
            shift = opt.shift
         }
         local filtersampler, filtersize = data.filtersizesampler{
            sizedataset = sizedataset,
            filter = filter
         }

         dataset = data.resample{
            dataset = dataset,
            sampler = filtersampler,
            size = filtersize
         }
         sizedataset = data.resample{
            dataset = sizedataset,
            sampler = filtersampler,
            size = filtersize
         }
         print('| batchresolution:', inputsizetransform(opt.samplerate/4))
         dataset = data.batch{
            dataset = data.resample{
               dataset = dataset,
               sampler = sampler
            },
            sizedataset = sizedataset,
            batchsize = opt.batchsize,
            batchresolution = inputsizetransform(opt.samplerate/4), -- 250ms
         }

         return dataset
      end
}

data.newiterator = argcheck{
   noordered = true,
   {name="closure", type="function"},
   {name="nthread", type="number"},
   call =
      function(closure, nthread)
         if nthread == 0 then
            return tnt.DatasetIterator{
               dataset = closure(),
            }
         else
            return tnt.ParallelDatasetIterator{
               closure = closure,
               nthread = nthread
            }
         end
      end
}

return data

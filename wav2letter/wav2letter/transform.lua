local argcheck = require 'argcheck'
local utils = require 'wav2letter.utils'
local transform = {}

transform.dictconvert = argcheck{
   noordered = true,
   {name="src", type="table"},
   {name="dst", type="table"},
   call =
      function(dictsrc, dictdst)
         local s2d = torch.LongTensor(#dictsrc)
         for s=1,#dictsrc do
            s2d[s] = assert(dictdst[dictsrc[s]], string.format("unknown token <%s> in destination dictionary", dictsrc[i]))
         end
         return
            function(src)
               return s2d:gather(1, src)
            end
      end
}

transform.pad = argcheck{
   {name="dim", type="number"},
   {name="size", type="number"},
   {name="value", type="number", default=0},
   call =
      function(dim, size, value)
         return
            function(src)
               assert(size == math.floor(size), 'size must be an integer')
               local sz = src:size()
               sz[dim] = sz[dim] + 2*size
               local dst = src.new(sz)
               dst:narrow(dim, 1, size):fill(value)
               dst:narrow(dim, size+1, src:size(dim)):copy(src)
               dst:narrow(dim, size+src:size(dim)+1, size):fill(value)
               return dst
            end
      end
}

transform.maxnormalize = argcheck{
   {name='threshold', type='number', default=0.01},
   call =
      function(threshold)
         return
            function(z)
               local max = math.max(
                  math.abs(z:min()),
                  math.abs(z:max())
               )
               z:add(-z:mean())
               if max > threshold then
                  z:div(max)
               end
               return z
            end
      end
}

transform.localnormalize = argcheck{
   {name='kw', type='number'},
   {name='dw', type='number'},
   {name='noisethreshold', type='number'},
   call =
      function(kw, dw, noisethresh)
         assert(math.floor(kw) == kw)
         assert(math.floor(dw) == dw)
         return function(x)
            local nchannel = x:size(2)
            local kw = kw*nchannel
            local dw = dw*nchannel
            local nthread = torch.getnumthreads()
            torch.setnumthreads(1)
            x = x:view(x:size(1)*nchannel)
            local xsz = x:size(1)
            if (xsz - kw) % dw ~= 0 then
               local nsz = math.ceil((xsz - kw) / dw)*dw + kw
               local z = x.new(nsz)
               z:narrow(1, 1, xsz):copy(x)
               z:narrow(1, xsz+1, nsz-xsz):zero()
               x = z
            end
            local c = x:clone():zero()
            c:unfold(1, kw, dw):add(1)
            local ux = x:unfold(1, kw, dw):clone()
            for i=1,ux:size(1) do
               local uxi = ux[i]
               uxi:add(-uxi:mean())
               local std = uxi:std()
               if std > noisethresh then
                  uxi:div(math.max(noisethresh, uxi:std()))
               end
            end
            local nx = x:clone():zero()
            nx:unfold(1, kw, dw):add(ux)
            nx:cdiv(c)
            nx = nx:narrow(1, 1, xsz)
            torch.setnumthreads(nthread)
            return nx:view(-1, nchannel)
         end
      end
}

transform.mostfrequentindex = argcheck{
   {name="kw", type="number"},
   {name="dw", type="number"},
   {name="win", type="number"},
   {name="nclass", type="number"},
   {name="padvalue", type="number", default=1},
   call =
      function(kw, dw, win, nclass, padvalue)
         local pad = transform.pad{
            dim=1,
            size=math.floor(kw/2),
            value=padvalue
         }
         return
            function(z)
               local nz = z.new()
               z = pad(z)
               z = z:unfold(1, kw, dw):narrow(2, math.floor((kw-1)/2)-(win-1)/2+1, win)
               utils.mostfrequentindex(nz, z, 2, nclass)
               nz = nz:squeeze(2)
               return nz
            end
      end
}

return transform

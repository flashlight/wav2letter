-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'
local errmsg = require 'gtn.errmsg'
local env = require 'gtn.env'
local C = env.C
local argcheck = require 'argcheck'

local GTN = {nodes=env.nodes, edges=env.edges}

GTN.new = argcheck{
   call =
      function()
         local self = C.GTNGraph_new()
         ffi.gc(self, C.GTNGraph_free)
         return self
      end
}

GTN.addNode = argcheck{
   {name='self', type='cdata'},
   {name='score', type='cdata'},
   {name='gradScore', type='cdata'},
   call =
      function(self, score, gradScore)
         local idx = tonumber(C.GTNGraph_addNode(self, score, gradScore))
         if idx < 0 then
            error(errmsg[idx])
         end
         return idx
      end
}

GTN.addEdge = argcheck{
   {name='self', type='cdata'},
   {name='srcidx', type='number'},
   {name='dstidx', type='number'},
   {name='score', type='cdata'},
   {name='gradScore', type='cdata'},
   call =
      function(self, srcidx, dstidx, score, gradScore)
         local idx = tonumber(C.GTNGraph_addEdge(self, srcidx, dstidx, score, gradScore))
         if idx < 0 then
            error(errmsg[idx])
         end
         return idx
      end
}

GTN.node = argcheck{
   {name='self', type='cdata'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         local score = ffi.new('float*[1]')
         local gradScore = ffi.new('float*[1]')
         local isActive = ffi.new('char[1]')
         C.GTNGraph_getNode(self, idx, score, gradScore, isActive)
         return score[0], gradScore[0], isActive[0] == 1
      end
}

GTN.dot = argcheck{
   {name='self', type='cdata'},
   {name='verbose', type='boolean', default=false},
   call =
      function(self, verbose)
         local txt = {'digraph ACN {'}
         table.insert(txt, 'rankdir=LR;')
         table.insert(txt, 'edge [penwidth=.3 arrowsize=0.8];')

         for idx, score, gradScore, isActive in self:nodes() do
            local label
            if verbose then
               label = string.format("%s\\n[%.4f] [%.4f]", idx, score[0], gradScore[0])
            else
               label = idx
            end
            table.insert(txt, string.format('id%s [label="%s" penwidth=.1 fontsize=10 style=filled fillcolor="%s"];', idx, label, isActive and "#eeeeee" or "#aaaaaa"))
         end

         for idx, srcidx, dstidx, score, gradScore in self:edges() do
            local label
            if verbose then
               label = string.format(' [label="[%.4f] [%.4f]"]', score[0], gradScore[0])
            else
               label = ""
            end
            table.insert(txt, string.format('id%s -> id%s%s;', srcidx, dstidx, label))
         end

         table.insert(txt, '}')
         txt = table.concat(txt, '\n')
         return txt
      end
}

GTN.nNode = argcheck{
   {name='self', type='cdata'},
   call =
      function(self)
         return tonumber(C.GTNGraph_nNode(self))
      end
}

GTN.nEdge = argcheck{
   {name='self', type='cdata'},
   call =
      function(self)
         return tonumber(C.GTNGraph_nEdge(self))
      end
}

GTN.forwardLogAdd = argcheck{
   {name='self', type='cdata'},
   {name='maxidx', type='number', default=-1},
   call =
      function(self, maxidx)
         return C.GTNGraph_forward_logadd(self, maxidx)
      end
}

GTN.backwardLogAdd = argcheck{
   {name='self', type='cdata'},
   {name='g', type='number'},
   {name='maxidx', type='number', default=-1},
   call =
      function(self, g, maxidx)
         C.GTNGraph_backward_logadd(self, g, maxidx)
      end
}

GTN.forwardMax = argcheck{
   {name='self', type='cdata'},
   {name='maxidx', type='number', default=-1},
   {name='path', type='cdata', opt=true},
   call =
      function(self, maxidx, path)
         if path then
            local size = ffi.new('long[1]')
            local score = C.GTNGraph_forward_max(self, maxidx, path, size)
            return score, tonumber(size[0])
         else
            return C.GTNGraph_forward_max(self, maxidx, nil, nil)
         end
      end
}

GTN.backwardMax = argcheck{
   {name='self', type='cdata'},
   {name='g', type='number'},
   {name='maxidx', type='number', default=-1},
   call =
      function(self, g, maxidx)
         C.GTNGraph_backward_max(self, g, maxidx)
      end
}

GTN.forward = argcheck{
   {name='self', type='cdata'},
   {name='maxidx', type='number', default=-1},
   {name='path', type='cdata', opt=true},
   {name='ismax', type='boolean', default=false},
   call =
      function(self, maxidx, path, ismax)
         if ismax then
            return self:forwardMax(maxidx, path)
         else
            return self:forwardLogAdd(maxidx)
         end
      end
}

GTN.backward = argcheck{
   {name='self', type='cdata'},
   {name='g', type='number'},
   {name='maxidx', type='number', default=-1},
   {name='ismax', type='boolean', default=false},
   call =
      function(self, g, maxidx, ismax)
         if ismax then
            return self:backwardMax(g, maxidx)
         else
            return self:backwardLogAdd(g, maxidx)
         end
      end
}

GTN.isActive = argcheck{
   {name='self', type='cdata'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 0 and idx <= self:nNode(), 'node index out of range')
         local isActive = C.GTNGraph_isActive(self, idx)
         return isActive == 1 and true or false
      end
}

ffi.metatype('GTNGraph', {__index=GTN})

env.GTN = {}
setmetatable(
   env.GTN,
   {
      __index = GTN,
      __newindx = GTN,
      __call = GTN.new
   })

return GTN

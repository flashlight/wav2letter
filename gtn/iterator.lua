-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'
local env = require 'gtn.env'
local C = env.C

function env.nodes(gtn)
   local iterator = C.GTNNodeIterator_new(gtn);
   local idx = ffi.new('long[1]')
   local score = ffi.new('float*[1]')
   local gradScore = ffi.new('float*[1]')
   local isActive = ffi.new('char[1]')
   return
      function()
         C.GTNNodeIterator_next(iterator, idx, score, gradScore, isActive);
         if idx[0] >= 0 then
            return tonumber(idx[0]), score[0], gradScore[0], (isActive[0] == 1 and true or false)
         end
      end
end

function env.edges(gtn, nodeidx)
   if nodeidx then
      local iterator = C.GTNNodeEdgeIterator_new(gtn, nodeidx);
      local srcidx = ffi.new('long[1]')
      local dstidx = ffi.new('long[1]')
      local score = ffi.new('float*[1]')
      local gradScore = ffi.new('float*[1]')
      return
         function()
            C.GTNNodeEdgeIterator_next(iterator, srcidx, dstidx, score, gradScore);
            if srcidx[0] >= 0 then
               return tonumber(srcidx[0]), tonumber(dstidx[0]), score[0], gradScore[0]
            end
         end
   else
      local iterator = C.GTNEdgeIterator_new(gtn);
      local idx = ffi.new('long[1]')
      local srcidx = ffi.new('long[1]')
      local dstidx = ffi.new('long[1]')
      local score = ffi.new('float*[1]')
      local gradScore = ffi.new('float*[1]')
      return
         function()
            C.GTNEdgeIterator_next(iterator, idx, srcidx, dstidx, score, gradScore);
            if idx[0] >= 0 then
               return tonumber(idx[0]), tonumber(srcidx[0]), tonumber(dstidx[0]), score[0], gradScore[0]
            end
         end
   end
end

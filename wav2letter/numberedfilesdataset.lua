-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local lfs = require 'lfs'

local NumberedFilesDataset =
   torch.class('tnt.NumberedFilesDataset', 'tnt.Dataset', tnt)

NumberedFilesDataset.__init = argcheck{
   {name='self', type='tnt.NumberedFilesDataset'},
   {name='path', type='string'},
   {name='features', type='table'},
   {name='maxload', type='number', opt=true},
   call =
      function(self, path, features, maxload)
         maxload = (maxload and maxload > 0) and maxload or math.huge
         self.__path = path
         self.__features = {}
         self.__aliases = {}
         self.__readers = {}
         assert(#features > 0, 'features should be a list of {name=<string>, reader=<function>, [alias=<string]}')
         for _, feature in ipairs(features) do
            assert(type(feature) == 'table'
                      and type(feature.name) == 'string'
                      and type(feature.reader) == 'function',
                   'features should be a list of {name=<string>, reader=<function>, [alias=<string]}')
            table.insert(self.__features, feature.name)
            table.insert(self.__readers, feature.reader)
            table.insert(self.__aliases, feature.alias or feature.name)
         end
         io.stderr:write(string.format("| dataset <%s>: ", path))
         io.stderr:flush()

         local pattern = "^(%d%d%d%d%d%d%d%d%d)%." .. features[1].name .. "$"
         local startidx
         for filename in lfs.dir(path) do
            if lfs.attributes(paths.concat(path, filename), "mode") == "file" then
               local idx = filename:match(pattern)
               if idx then
                  idx = tonumber(idx)
                  startidx = startidx or idx
                  startidx = math.min(startidx, idx)
               end
            end
         end
         if not startidx then
            startidx = 0
            self.__subdir = true
         end
         self.__startidx = startidx
         for i=1,maxload do
            local f = io.open(
               self:filename(i, features[1].name)
            )
            if f then
               f:close()
            else
               self.__size = i-1
               break
            end
         end
         self.__size = self.__size or maxload
         io.stderr:write(string.format("%d files found\n", self.__size))
         io.stderr:flush()
         assert(self.__size > 0, string.format("no file found in <%s/?????????.%s> nor in <%s/00000/?????????.%s>", path, features[1].name, path, features[1].name))
      end
}

NumberedFilesDataset.filename = argcheck{
   {name='self', type='tnt.NumberedFilesDataset'},
   {name='idx', type='number'},
   {name='feature', type='string'},
   call =
      function(self, idx, feature)
         local path = self.__path
         if self.__subdir then
            path = paths.concat(
               path,
               string.format('%05d', math.floor((idx-1) / 10000))
            )
         end
         return paths.concat(
            path,
            string.format("%09d.%s", idx-1+self.__startidx, feature)
         )
      end
}

NumberedFilesDataset.size = argcheck{
   {name='self', type='tnt.NumberedFilesDataset'},
   call =
      function(self)
         return self.__size
      end
}

NumberedFilesDataset.get = argcheck{
   {name='self', type='tnt.NumberedFilesDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         local sample = {}
         for i, feature in ipairs(self.__features) do
            local alias = self.__aliases[i]
            local reader = self.__readers[i]
            sample[alias] = reader(
               self:filename(idx, feature)
            )
         end
         return sample
      end
}

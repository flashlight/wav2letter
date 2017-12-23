-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'torch'
require 'paths'
require 'gnuplot'

local function usage(arg)
   print(string.format("usage: %s [options] <column name> <files>", arg[0]))
   print[[
options:
  %web              -- output path root is $HOME/public_html
  %pdf[=<filename>] -- pdf output (optional filename) default: plot.pdf
  %png[=<filename>] -- png output (optional filename) default: plot.png
  %svg[=<filename>] -- svg output (optional filename) default: plot.svg
  %xmin=<value>     -- xmin value
  %xmax=<value>     -- xmax value
  %ymin=<value>     -- ymin value
  %ymax=<value>     -- ymax value
  +<key>=<value>    -- select only experiments where config <key> is equal to <value>
  -<key>=<value>    -- filter out  experiments where config <key> is equal to <value>
]]
   return false
end

local opt = {outpath='.'}

local invalidkeys = {
   gfsai=true, rundir=true, netspecs=true, dataspecs=true, kw=true,
   dw=true, force=true, timestamp=true, reload=true, path=true,
   hostname=true, datadir=true, archdir=true, gpu=true, iter=true,
   username=true, nclass=true
}

local function convert(value, typename)
   if typename == 'boolean' then
      if value == 'false' then
         value = false
      elseif value == 'true' then
         value = true
      end
   elseif typename == 'number' then
      value = tonumber(value)
   end
   return value
end

local title = {}
local titleflag
for i=1,#arg do
   local z = arg[i]
   if z:match('^%%')
      or z:match('^%+')
      or z:match('^%-') then
         table.insert(title, z)
   elseif not titleflag then
      titleflag = true
      table.insert(title, z)
   end
end
table.insert(title, '...')
title = table.concat(title, ' ')

local filters = {}
for i=#arg,1,-1 do
   local z = arg[i]
   if z:match('^%%pdf') then
      local filename = z:match('^%%pdf%=(.+)') or 'plot.pdf'
      opt.pdf = filename
      table.remove(arg, i)
   elseif z:match('^%%png') then
      local filename = z:match('^%%png%=(.+)') or 'plot.png'
      opt.png = filename
      table.remove(arg, i)
   elseif z:match('^%%svg') then
      local filename = z:match('^%%svg%=(.+)') or 'plot.svg'
      opt.svg = filename
      table.remove(arg, i)
   elseif z:match('^%%web') then
      local subpath = z:match('^%%web%=(.+)')
      if subpath then
         opt.outpath = paths.concat(os.getenv('HOME'), 'public_html', subpath)
      else
         opt.outpath = paths.concat(os.getenv('HOME'), 'public_html')
      end
      table.remove(arg, i)
   elseif z:match('^%%ymin%=') then
      opt.ymin = tonumber(z:match('^%%ymin%=(.+)'))
      table.remove(arg, i)
   elseif z:match('^%%ymax%=') then
      opt.ymax = tonumber(z:match('^%%ymax%=(.+)'))
      table.remove(arg, i)
   elseif z:match('^%%xmin%=') then
      opt.xmin = tonumber(z:match('^%%xmin%=(.+)'))
      table.remove(arg, i)
   elseif z:match('^%%xmax%=') then
      opt.xmax = tonumber(z:match('^%%xmax%=(.+)'))
      table.remove(arg, i)
   elseif z:match('^%+.+%=.+') then
      local key, value = z:match('^%+(.+)%=(.+)')
      table.insert(
         filters,
         function(config)
            if config[key] == convert(value, type(config[key])) then
               return true
            end
         end
      )
      table.remove(arg, i)
   elseif z:match('^%-.+%=.+') then
      local key, value = z:match('^%-(.+)%=(.+)')
      table.insert(
         filters,
         function(config)
            if config[key] ~= convert(value, type(config[key])) then
               return true
            end
         end
      )
      table.remove(arg, i)
   end
end

local column = arg[1]
local ylabel = column
if not column then
   assert(usage(arg))
end

-- escapes
column = column:gsub('([%(%)%.%+%-%*%?%[%]%^%$%%])',
                     function(str)
                        return '%' .. str
                     end)
table.remove(arg, 1)

local function loadfile(filename)
   local x = {}
   local y = {}
   local x_ = 0
   pcall(
      function()
         for line in io.lines(filename) do
            if not line:match(column .. '^%s*#') then
               local y_ = tonumber(line:match(column .. '%s*([^%s|]+)'))
               if y_ then
                  x_ = x_ + 1
                  table.insert(x, x_)
                  table.insert(y, y_)
               end
            end
         end
      end
   )
   collectgarbage() -- io.lines()
   return torch.Tensor(x), torch.Tensor(y)
end

local files = {}

local avlkeys = {}
for i=1,#arg do
   local f
   if pcall(
      function()
         f = torch.DiskFile(paths.concat(arg[i], 'model-last.bin')):binary()
      end) then
      f:readObject()
      local config = f:readObject()
      f:close()

      local accept = true
      for _, filter in ipairs(filters) do
         if not filter(config) then
            accept = false
            break
         end
      end

      if accept then
         for k,v in pairs(config) do
            if not invalidkeys[k] then
               avlkeys[k] = avlkeys[k] or {}
               avlkeys[k][v] = avlkeys[k][v] or 1
               avlkeys[k][v] = avlkeys[k][v] + 1
            end
         end

         table.insert(files, {path=arg[i], config=config})
      end
   end
end

local keys = {}
for k,v in pairs(avlkeys) do
   local valid = false
   local n = 0
   local sz = 0
   for k,v in pairs(v) do
      n = n + 1
      sz = math.max(sz, #tostring(k))
   end
   if n == 1 then
      for k,v in pairs(v) do
         if v == 1 then
            valid = true
         end
      end
   else
      valid = true
   end
   if valid then
      table.insert(keys, {name=k, size=sz})
   end
end
table.sort(keys, function(a, b) return a.name < b.name end)

local function legend(config, keys)
   local legend = {}
   for _,k in ipairs(keys) do
      local value = tostring(config[k.name])
      value = value .. string.rep(' ', k.size-#value)
      table.insert(legend, string.format("%s=%s", k.name, value))
   end
   return table.concat(legend, ' ')
end

local plots = {}
local xmin = math.huge
local xmax = -math.huge
local ymin = math.huge
local ymax = -math.huge
for _, file in ipairs(files) do
   local x, y = loadfile(paths.concat(file.path, 'perf'))
   if x:nDimension() > 0 and x:size(1) > 0 then
      xmin = math.min(xmin, x:min())
      xmax = math.max(xmax, x:max())
      ymin = math.min(ymin, y:min())
      ymax = math.max(ymax, y:max())
      table.insert(
         plots,
         {
            legend(file.config, keys),
            x,
            y,
            '+-'
         }
      )
   end
end

if opt.pdf then
   os.execute(string.format('mkdir -p "%s"', opt.outpath))
   gnuplot.pdffigure(paths.concat(opt.outpath, opt.pdf))
   gnuplot.raw('set terminal pdfcairo enhanced size 35cm,25cm')
   gnuplot.raw('set key font "Monospace,6"')
elseif opt.png then
   os.execute(string.format('mkdir -p "%s"', opt.outpath))
   gnuplot.pngfigure(paths.concat(opt.outpath, opt.png))
   gnuplot.raw('set terminal pngcairo enhanced size 1400,1050')
   gnuplot.raw('set key font "Monospace,8"')
elseif opt.svg then
   os.execute(string.format('mkdir -p "%s"', opt.outpath))
   gnuplot.svgfigure(paths.concat(opt.outpath, opt.svg))
   gnuplot.raw('set terminal svg enhanced dynamic size 1400,1050')
   gnuplot.raw('set key font "Monospace,8"')
else
   gnuplot.raw('set term wxt enhanced size 1400,1050')
   gnuplot.raw('set key font "Monospace,8"')
end
gnuplot.raw('set key outside tmargin')

gnuplot.xlabel("epoch")
gnuplot.ylabel(ylabel)
gnuplot.title(title)
gnuplot.axis({opt.xmin or xmin, opt.xmax or xmax, opt.ymin or ymin, opt.ymax or ymax})
gnuplot.plot(plots)
gnuplot.grid(true)

if opt.pdf or opt.png or opt.svg then
   gnuplot.plotflush()
end

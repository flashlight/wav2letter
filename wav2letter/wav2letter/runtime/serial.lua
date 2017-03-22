local argcheck = require 'argcheck'
local md5 = require 'md5'
local serial = {}
local ffi = require 'ffi'

ffi.cdef[[
int symlink(const char *path1, const char *path2);
]]

serial.symlink = argcheck{
   {name="dir", type="string"},
   {name="link", type="string"},
   call =
      function(dir, link)
         return ffi.C.symlink(dir, link) == 0
      end
}

serial.savecmdline = argcheck{
   noordered = true,
   {name="arg", type="table"},
   call =
      function(arg)
         local str = {}
         for i=0,#arg do
            table.insert(str, string.format("%q", arg[i]))
         end
         return table.concat(str, " ")
      end
}

serial.cleanfilename = argcheck{
   {name="filename", type="string"},
   call =
      function(filename)
         filename = filename:gsub('[^%w_%-%.]', '#')
         return filename
      end
}

serial.loadmodel = argcheck{
   {name="filename", type="string"},
   {name="arch", type="boolean", default=false},
   call =
      function(filename, larch)
         local model = {}
         local f = torch.DiskFile(filename):binary()
         model.config = f:readObject()
         if larch then
            model.arch = f:readObject()
         end
         f:close()
         return model
      end
}

serial.savemodel = argcheck{
   {name="filename", type="string"},
   {name="config", type="table"},
   {name="arch", type="table"},
   call =
      function(filename, config, arch)
         local f = torch.DiskFile(filename, 'w')
         f:binary()
         f:writeObject(config)
         f:writeObject(arch)
         f:close()
      end
}

serial.newpath = argcheck{
   {name="root", type="string"},
   {name="opt", type="table"},
   call =
      function(root, opt)
         local path = {}
         if opt.runname == '' then
            table.insert(path, os.date("%Y-%m-%d_%H-%M-%S"))
            table.insert(path, os.getenv('HOSTNAME'))
            table.insert(path, md5.sumhexa(serial.savetable{tbl=opt}))
         else
            table.insert(path, opt.runname)
         end
         if opt.tag ~= '' then
            table.insert(path, opt.tag)
         end
         return paths.concat(root, table.concat(path, '_'))
      end
}

serial.parsecmdline = argcheck{
   {name="closure", type="function"},
   {name="arg", type="table"},
   {name="default", type="table", opt=true},
   call =
      function(closure, arg, default)
         local cmd = torch.CmdLine()
         if default then
            local oldcmdoption = cmd.option
            function cmd:option(name, def, help)
               local rawname = name:gsub("^%-+", "")
               if default[rawname] ~= nil then
                  def = default[rawname]
               end
               oldcmdoption(cmd, name, def, help)
            end
         end
         closure(cmd)
         local opt = cmd:parse(arg)
         if default then
            -- add missing options
            for k, v in pairs(default) do
               if opt[k] == nil then
                  opt[k] = default[k]
               end
            end
         end
         return opt
      end
}

serial.mkdir = argcheck{
   {name="path", type="string"},
   call =
      function(path)
         if not paths.dirp(path) then
            os.execute(string.format('mkdir -p %q', path))
         end
      end
}

serial.runidx = argcheck{
   {name="path", type="string"},
   {name="filename", type="string"},
   {name="idx", type="number", opt=true},
   call =
      function(path, filename, idx)
         if idx then
            return paths.concat(path, string.format("%03d_%s", idx, filename))
         else
            for idx=1,999 do
               local f = io.open(paths.concat(path, string.format("%03d_%s", idx, filename)))
               if not f then
                  return idx
               else
                  f:close()
               end
            end
         end
      end
}

serial.loadtable = argcheck{
   {name="filename", type="string"},
   call =
      function(filename)
         local f = torch.DiskFile(filename, "r")
         local str = f:readString("*all")
         f:close()
         local opt, err = loadstring(str)
         if not opt then
            error(string.format("could not parse lua file <%s> [%s]", filename, err))
         end
         opt = opt()
         return opt
      end
}

local function table2string(tbl)
   local str = {"{"}
   local keys = {}
   for k, v in pairs(tbl) do
      table.insert(keys, k)
   end
   table.sort(keys)
   for _, k in ipairs(keys) do
      local v = tbl[k]
      assert(type(k) == "string")
      if type(v) == "number" or type(v) == "boolean" then
         table.insert(str, string.format("%s = %s,", k, v))
      elseif type(v) == "string" then
         table.insert(str, string.format("%s = %q,", k, v))
      elseif type(v) == "table" then
         table.insert(str, string.format("%s = %s,", k, table2string(v)))
      else
         error("invalid table value type (number, string or boolean expected)")
      end
   end
   table.insert(str, "}")
   return table.concat(str, "\n")
end

serial.savetable = argcheck{
   noordered = true,
   {name="filename", type="string", opt=true},
   {name="tbl", type="table"},
   call=
      function(filename, tbl)
         local str = table2string(tbl)
         if filename then
            local f = torch.DiskFile(filename, "w")
            f:writeString("return ")
            f:writeString(str)
            f:close()
         end
         return str
      end
}

serial.heartbeat = argcheck{
   noordered = true,
   {name="filename", type="string", opt=true},
   {name="interval", type="number", default=60},
   {name="closure", type="function", opt=true},
   call=
      function(filename, interval, closure)
         local clock
         return function(...)
            local newclock = os.clock()
            if not clock or newclock-clock > interval then
               local f = assert(
                  io.open(filename, "w"),
                  string.format("could not open <%s> for writing", filename)
               )
               f:close()
               clock = newclock
               if closure then
                  closure(filename, ...)
               end
            end
         end
      end
}

return serial

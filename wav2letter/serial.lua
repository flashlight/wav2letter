local argcheck = require 'argcheck'
local md5 = require 'md5'
local serial = {}

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
         model.info = f:readObject()
         if larch then
            model.arch = f:readObject()
         end
         f:close()
         return model
      end
}

serial.savemodel = argcheck{
   {name="filename", type="string"},
   {name="info", type="table"},
   {name="arch", type="table"},
   call =
      function(filename, info, arch)
         local f = torch.DiskFile(filename, 'w')
         f:binary()
         f:writeObject(info)
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
         table.insert(path, os.date("%Y-%m-%d_%H-%M-%S"))
         table.insert(path, os.getenv('HOSTNAME'))
         if opt.tag ~= '' then
            table.insert(path, opt.tag)
         end
         table.insert(path, md5.sumhexa(serial.saveopt{opt=opt}))
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
         os.execute(string.format('mkdir -p "%s"', path))
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

serial.loadopt = argcheck{
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

serial.saveopt = argcheck{
   noordered = true,
   {name="filename", type="string", opt=true},
   {name="opt", type="table"},
   call=
      function(filename, opt)
         local str = {"return {"}
         for k, v in pairs(opt) do
            assert(type(k) == "string")
            if type(v) == "number" or type(v) == "boolean" then
               table.insert(str, string.format("%s = %s,", k, v))
            elseif type(v) == "string" then
               table.insert(str, string.format("%s = '%s',", k, v))
            else
               error("invalid opt value type (number, string or boolean expected)")
            end
         end
         table.insert(str, "}\n")
         str = table.concat(str, "\n")
         if filename then
            local f = torch.DiskFile(filename, "w")
            f:writeString(str)
            f:close()
         end
         return str
      end
}

return serial

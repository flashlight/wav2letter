local sndfile = require 'sndfile'
local argcheck = require 'argcheck'

local readers = {}

readers.audio = argcheck{
   noordered = true,
   {name='samplerate', type='number', opt=true},
   {name='channels', type='number', opt=true},
   call =
      function(samplerate, channels)
         return function(filename)
            local f = assert(
               sndfile.SndFile(filename),
               string.format("could not read file <%s>", filename))
            local info = f:info()
            if samplerate and info.samplerate ~= samplerate then
               error("NYI: sample rate conversion")
            end
            if channels and info.channels ~= channels then
               error("NYI: # of channel conversion")
            end
            local data = f:readFloat(info.frames)
            f:close()
            return data
         end
      end
}

readers.tokens = argcheck{
   noordered = true,
   {name='dictionary', type='table'},
   call =
      function(dictionary)
         return function(filename)
            local f = assert(
               io.open(filename),
               string.format("could not read file <%s>", filename))
            local data = f:read('*all')
            f:close()
            local tokens = {}
            for token in data:gmatch('(%S+)') do
               local tokenidx = dictionary[token]
               if not tokenidx then
                  error(string.format("token <%s> not in the dictionary", token))
               end
               table.insert(tokens, tokenidx)
            end
            return torch.LongTensor(tokens)
         end
      end
}

readers.number = argcheck{
   call =
      function()
         return function(filename)
            local f = assert(
               io.open(filename),
               string.format("could not read file <%s>", filename))
            local number = tonumber(f:read('*line'))
            assert(number, "parsing error: no number in file")
            f:close()
            return number
         end
      end
}

return readers

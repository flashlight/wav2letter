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
            local data = f:readFloat(info.frames)
            if samplerate and info.samplerate ~= samplerate then
               local sampleratelib = require 'samplerate'
               local samplingRate = samplerate / info.samplerate
               data = sampleratelib.sample(data, samplingRate)
            end
            if channels and info.channels ~= channels then
               error("NYI: # of channel conversion")
            end
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

readers.words = argcheck{
   noordered=true,
   {name="dict", type="tds.Hash", opt=true},
   call =
      function(dictionary)
         local unkidx = dictionary["<unk>"]
         return function(filename)
            local f = io.open(filename)
            local data
            if f then
               data = f:read('*all')
               f:close()
            end
            if dictionary then
               local tokens = {}
               for token in data:gmatch('(%S+)') do
                  local tokenidx = dictionary[token] and dictionary[token] or unkidx
                  table.insert(tokens, tokenidx)
               end
               data = torch.LongTensor(tokens)
            end
            return data
         end
      end
}

return readers

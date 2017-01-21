local tnt = require 'torchnet'
local argcheck = require 'argcheck'

local SpeechStatMeter = torch.class('tnt.SpeechStatMeter', 'tnt.Meter', tnt)

SpeechStatMeter.__init = argcheck{
   {name="self", type="tnt.SpeechStatMeter"},
   call =
      function(self)
         self:reset()
      end
}

SpeechStatMeter.reset = argcheck{
   {name="self", type="tnt.SpeechStatMeter"},
   call =
      function(self)
         self.isz = 0
         self.tsz = 0
         self.maxisz = 0
         self.maxtsz = 0
         self.n = 0
      end
}

SpeechStatMeter.add = argcheck{
   {name="self", type="tnt.SpeechStatMeter"},
   {name="input", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call =
      function(self, input, target)
         local isz = input:size(1)
         local tsz = target:size(1)
         self.isz = self.isz + isz
         self.tsz = self.tsz + tsz
         self.maxisz = math.max(self.maxisz, isz)
         self.maxtsz = math.max(self.maxtsz, tsz)
         self.n = self.n + 1
      end
}

SpeechStatMeter.value = argcheck{
   {name="self", type="tnt.SpeechStatMeter"},
   {name="name", type="string", opt=true},
   call =
      function(self, name)
         if name == nil then
            return {
               isz = self.isz,
               tsz = self.tsz,
               maxisz = self.maxisz,
               maxtsz = self.maxtsz,
               n = self.n
            }
         else
            return self[name] or error('invalid name (isz, tsz, maxisz, maxtsz or n expected)')
         end
      end
}

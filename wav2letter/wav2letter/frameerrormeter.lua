local tnt = require 'fbtorchnet'
local argcheck = require 'argcheck'

local FrameErrorMeter = torch.class('tnt.FrameErrorMeter', 'tnt.Meter', tnt)

FrameErrorMeter.__init = argcheck{
   {name="self", type="tnt.FrameErrorMeter"},
   {name="accuracy", type="boolean", default=false},
   call =
      function(self, accuracy)
         self.accuracy = accuracy
         self:reset()
      end
}

FrameErrorMeter.reset = argcheck{
   {name="self", type="tnt.FrameErrorMeter"},
   call =
      function(self)
         self.sum = 0
         self.n = 0
      end
}

FrameErrorMeter.add = argcheck{
   {name="self", type="tnt.FrameErrorMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call =
      function(self, output, target)
         assert(target:nDimension() == 1, 'target: vector expected')
         assert(output:nDimension() == 1, 'output: vector expected')
         assert(
            target:size(1) == output:size(1),
            'target and output do not match')

         local sum = 0
         local n = output:size(1)
         for i=1,n do
            if output[i] ~= target[i] then
               sum = sum + 1
            end
         end
         self.sum = self.sum + sum
         self.n = self.n + n
      end
}

FrameErrorMeter.value = argcheck{
   {name="self", type="tnt.FrameErrorMeter"},
   call =
      function(self)
         local value = self.sum / self.n * 100
         if self.accuracy then
            return 100 - value
         else
            return value
         end
      end
}

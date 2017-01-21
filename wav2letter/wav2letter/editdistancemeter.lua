local tnt = require 'fbtorchnet'
local argcheck = require 'argcheck'
local utils = require 'wav2letter.utils'

local EditDistanceMeter = torch.class('tnt.EditDistanceMeter', 'tnt.Meter', tnt)

EditDistanceMeter.__init = argcheck{
   {name="self", type="tnt.EditDistanceMeter"},
   call =
      function(self)
         self:reset()
      end
}

EditDistanceMeter.reset = argcheck{
   {name="self", type="tnt.EditDistanceMeter"},
   call =
      function(self)
         self.sum = 0
         self.n = 0
      end
}

EditDistanceMeter.add = argcheck{
   {name="self", type="tnt.EditDistanceMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call =
      function(self, output, target)
         assert(target:nDimension() <= 1, 'target: vector expected')
         assert(output:nDimension() <= 1, 'output: vector expected')
         self.sum = self.sum + utils.editdistance(output:long(), target:long())
         self.n = self.n + (target:nDimension() > 0 and target:size(1) or 0)
      end
}

EditDistanceMeter.value = argcheck{
   {name="self", type="tnt.EditDistanceMeter"},
   call =
      function(self)
         return self.sum / self.n * 100
      end
}

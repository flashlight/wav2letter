require 'cunn'
local DataParallelTableTable, parent = torch.class('nn.DataParallelTableTable', 'nn.DataParallelTable')

function DataParallelTableTable:__init(flattenParams, usenccl)
   parent.__init(self, 0, flattenParams, usenccl)
   self.output = {}
   self.gradInput = nil
end

function DataParallelTableTable:_distribute(dst, src)
   for i = 1, #self.gpuAssignments do
      if src[i] then
         dst[i] = src[i]
      else
         dst[i] = torch.Tensor()
      end
   end
end

function DataParallelTableTable:_concat(dst, src)
   for i = 1, #self.gpuAssignments do
      dst[i] = src[i]
   end
   return dst
end

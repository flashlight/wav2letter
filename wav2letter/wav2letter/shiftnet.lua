local ShiftNet, Container = torch.class('nn.ShiftNet', 'nn.Container')

function ShiftNet:__init(network, shift)
   Container.__init(self)
   self.network = network
   self.modules = network.modules
   self.shift = shift
   self.output = torch.Tensor()
   self.gradOutput = {}
   for i=1,self.shift do
      self.gradOutput[i] = torch.Tensor()
   end
end

function ShiftNet:updateOutput(input)
   assert(#input == self.shift)
   local outputs = self.network:updateOutput(input)
   local n = 0
   for i=1,self.shift do
      n = n + outputs[i]:size(1)
   end
   self.output:resize(n, outputs[1]:size(2)) -- 2D only for now
   for i=1,self.shift do
      self.output:narrow(
         1, i, self.output:size(1)-i+1):unfold(
         1, 1, self.shift):copy(outputs[i])
   end
   return self.output
end

function ShiftNet:updateGradInput(input, gradOutput)
   -- we store a temp copy to avoid redoing the same thing in accGradParameters
   for i=1,self.shift do
      self.gradOutput[i]:resizeAs(self.network.output[i]):copy(
         gradOutput:narrow(
            1, i, gradOutput:size(1)-i+1):unfold(
            1, 1, self.shift))
   end
   self.gradInput = self.network:updateGradInput(input, self.gradOutput)
   return self.gradInput
end

function ShiftNet:accGradParameters(input, gradOutput)
   self.network:accGradParameters(input, self.gradOutput)
end

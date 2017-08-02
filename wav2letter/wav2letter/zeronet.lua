local ZeroNet, Module = torch.class('nn.ZeroNet', 'nn.Module')

function ZeroNet:__init(kw, dw, nclass)
   Module.__init(self)
   self.kw = kw
   self.dw = dw
   self.nclass = nclass
end


function ZeroNet:parameters()
    return {}, {}
end

function ZeroNet:updateOutput(input)
   local T = input:size(1)
   T = math.floor((T-self.kw)/self.dw)+1
   self.output:resize(T, self.nclass):zero()
   return self.output
end

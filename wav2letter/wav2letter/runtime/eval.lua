require 'nn'
require 'cunn'
require 'cudnn'
local argcheck = require 'argcheck'
local serial = require 'wav2letter.runtime.serial'
local transf = require 'wav2letter.runtime.transforms'

local eval = {}
eval.loadModel = argcheck{
    {name="filename", type="string"},
    call =
        function(filePath)
            local model = serial.loadmodel{filename=filePath, arch=true}
            local opt  = model.config.opt
            --disable train mode in the network
            model.arch.network:evaluate()
            local transforms = transf.inputfromoptions(opt,
                                                       model.config.kw,
                                                       model.config.dw)
            return model.arch.network, transforms
        end
}

eval.getModelDepth = argcheck{
    {name="network", type="nn*"},
    call=
        function(network)
            return network:size()
        end
}

eval.feedForward = argcheck{
  {name="network", type="nn*"},
  {name="transforms", type="function"},
  {name="input", type="torch.FloatTensor"},
  {name="depth", type="number", opt=true},
  call =
     function(network, transforms, input, depth)
        network:forward(transforms(input))
        local output = network:get(depth or network:size()).output:clone()
        local totalSize = output:size(1)
        local outputSizes = output:size()
        for i = 2, #outputSizes do
            totalSize = totalSize * outputSizes[i]
        end
        return output, totalSize
     end
}

return eval

-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local argcheck = require 'argcheck'
local optim = {}

optim.normGradientClamp = argcheck{
    doc = [[
    <a name="optim.normGradientClamp">
    #### optim.normGradientClamp(@ARGP)
    @ARGT

    This function clamps the norm of the gradient to the provided value.

    This function should be called before `updateParameters()`.

    Example:
    ```
    > optim = require 'torchnet.optim'
    > net = nn.Linear(10,10)
    > applyClamp = optim.normGradientClamp(net)
    > applyClamp(0.9)
    ```
    ]],
    {name = 'network', type = 'nn.Module'},
    {name = 'criterion', type = 'nn.Criterion', opt=true},
    call = function(network, criterion)
       local _, dw = network:parameters()
       local _, cdw = criterion and criterion:parameters()
       cdw = cdw or {}

       return function(maxnorm)
          assert(maxnorm and type(maxnorm) == 'number')
          assert(maxnorm > 0)

          local norm = 0
          for i, g in ipairs(dw) do
             local z = g:norm()
             norm = norm + z*z
          end
          for i, g in ipairs(cdw) do
             local z = g:norm()
             norm = norm + z*z
          end
          norm = math.sqrt(norm)

          if norm > maxnorm then
             local scale = maxnorm/norm
             for i, g in ipairs(dw) do
                g:mul(scale)
             end
             for i, g in ipairs(cdw) do
                g:mul(scale)
             end
          end
        end
    end
}

optim.weightedGradientClamp = argcheck{
    doc = [[
    <a name="optim.weightedGradientClamp">
    #### optim.weightedGradientClamp(@ARGP)
    @ARGT

   This function clamps the gradient g (in each dimension) such that
   -(scale*|w|+value) <= g <= (scale*|w|+value),
   where w is the corresponding weight value.

    This function should be called before `updateParameters()`.

    Example:
    ```
    > optim = require 'torchnet.optim'
    > net = nn.Linear(10,10)
    > applyClamp = optim.weightedGradientClamp(net)
    > applyClamp(0.9, 0.001)
    ```
    ]],
    {name = 'network', type = 'nn.Module'},
    {name = 'criterion', type = 'nn.Criterion', opt=true},
    call = function(network, criterion)
       local function init(w, dw)
          local mask = {}
          local saw = {}
          local byte = {}

          -- init masks and byte
          for i, g in ipairs(dw) do
             saw[i] = w[i]:clone()
             mask[i] = g.new(g:size()):zero()
             byte[i] = torch.type(g) == 'torch.CudaTensor' and mask[i] or
                torch.ByteTensor(g:size())
          end

          return function(maxval, scale)
             assert(maxval and type(maxval) == 'number' and maxval > 0)
             assert(scale and type(scale) == 'number' and scale > 0)

             for i, g in ipairs(dw) do
                local saw = saw[i]
                local mask = mask[i]
                local byte = byte[i]
                local w = w[i]
                saw:copy(w):abs():mul(scale):add(maxval)
                mask:gt(g, saw)
                local byte = torch.type(g) == 'torch.CudaTensor' and mask
                   or byte:copy(mask)
                g[byte] = saw[byte]
                saw:copy(w):abs():mul(-scale):add(-maxval)
                mask:lt(g, saw)
                byte = torch.type(g) == 'torch.CudaTensor' and mask
                   or byte:copy(mask)
                g[byte] = saw[byte]
             end
          end
       end

       local applynet = init(network:parameters())
       local applycrt = criterion
          and init(criterion:parameters())
          or function() end

       return function(maxval, scale)
          applynet(maxval, scale)
          applycrt(maxval, scale)
       end
    end
}

optim.absGradientClamp = argcheck{
    doc = [[
    <a name="optim.absGradientClamp">
    #### optim.absGradientClamp(@ARGP)
    @ARGT

    The absolute value (in each dimension) of the gradient will be clamped
    to the provided value.

    This function should be called before `updateParameters()`.

    Example:
    ```
    > optim = require 'torchnet.optim'
    > net = nn.Linear(10,10)
    > applyClamp = optim.absGradientClamp(net)
    > applyClamp(0.9)
    ```
    ]],
    {name = 'network', type = 'nn.Module'},
    {name = 'criterion', type = 'nn.Criterion', opt=true},
    call = function(network, criterion)
       local function init(_, dw)
          local mask = {}
          local byte = {}
          -- init masks and byte
          for i, g in ipairs(dw) do
             mask[i] = g.new(g:size()):zero()
             byte[i] = torch.type(g) == 'torch.CudaTensor' and mask[i] or
                torch.ByteTensor(g:size())
          end

          -- apply clamp
          return function(maxval)
             assert(maxval and type(maxval) == 'number' and maxval > 0)
             for i, g in ipairs(dw) do
                local mask = mask[i]
                local byte = byte[i]
                mask:gt(g, maxval)
                local byte = torch.type(g) == 'torch.CudaTensor' and mask
                   or byte:copy(mask)
                g[byte] = maxval
                mask:lt(g, -maxval)
                byte = torch.type(g) == 'torch.CudaTensor' and mask
                   or byte:copy(mask)
                g[byte] = -maxval
             end
          end
       end

       local applynet = init(network:parameters())
       local applycrt = criterion
          and init(criterion:parameters())
          or function() end

       return function(maxval)
          applynet(maxval)
          applycrt(maxval)
        end
    end
}

return optim

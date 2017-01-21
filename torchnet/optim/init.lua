local argcheck = require 'argcheck'
local doc      = require 'argcheck.doc'

doc[[

### tnt.optim

*Torchnet* provides a set of general optimization related functions.
These functions must be call in the engine.

]]

local optim = {}

optim.weightDecay = argcheck{
    doc = [[
        <a name="optim.weightDecay">
        #### optim.weightDecay(@ARGP)
        @ARGT

        This function implements the weight decay for a given list of modules.
        Weight decay is equivalent to a L2 norm regularization of the
        parameters.

        This function creates a closure which can later be called during the
        optimization.  This function should be called before
        `updateParameters()`.

        For example:
        ```
        > optim = require 'torchnet.optim'
        > net = nn.Linear(10,10)
        > modules = net:listModules()
        > applyWeightDecay = optim.weightDecay(modules)
        > applyWeightDecay(0.01)
        ```
        will apply a weight decay of 0.01 to the network `net`.
    ]],
    {name = 'modules', type = 'table'},
    call = function(modules)
        return function(decay)
            assert(decay >= 0, 'decay should be positive!')
            if decay == 0 then return end
            for _, m in pairs(modules) do
                if m.weight then m.weight:mul(1 - decay) end
            end
        end
    end
}

optim.weightDecay = argcheck{
    doc = [[
    <a name="optim.weightDecay_2">
    #### optim.weightDecay(@ARGP)
    @ARGT

    Same as [weightDecay](#optim.weightDecay) but on a network instead of a
    list of modules.

    This function can take as an option a list of module types to ignore.  By
    default this list is restricted to batch normalization modules.

    This function should be called before `updateParameters()`.

    Example:
    ```
    > optim = require 'torchnet.optim'
    > net = nn.Linear(10,10)
    > applyWeightDecay = optim.weightDecay(net)
    > applyWeightDecay(0.01)
    ```
    ]],
    {name = 'network',       type = 'nn.Module'},
    {name = 'skip',   type = 'table', opt = true},
    overload = optim.weightDecay,
    call = function(network, skip)
        skip = skip or {'nn.BatchNormalization','nn.SpatialBatchNormalization'}
        local modules = network:listModules()
        for i, m in pairs(modules) do
            for _, t in pairs(skip) do
                if torch.isTypeOf(m, t) then modules[i] = nil end
            end
        end
        return optim.weightDecay(modules)
    end
}

---- function that implements momentum:
-- create a function which should be run *before* updateParameters()
optim.momentum = argcheck{
    doc = [[
    <a name="optim.momentum">
    #### optim.momentum(@ARGP)
    @ARGT

    This function implements the momentum for a given network. It creates a
    closure around the network which can later be called during optimization.

    This function should be called before `updateParameters()`.

    Example:
    ```
    > optim = require 'torchnet.optim'
    > net = nn.Linear(10,10)
    > applyMomentum = optim.momentum(net)
    > applyMomentum(0.9)
    ```
    ]],
    {name = 'network', type = 'nn.Module'},
    call = function(network)
        local dm    = {}
        local _, dw = network:parameters()
        for i, g in ipairs(dw) do
            dm[i] = g.new(g:size()):zero()
        end
        return function(mom)
            assert(mom and type(mom) == 'number' and mom >= 0)
            if mom == 0 then return dw end
            for i, g in ipairs(dw) do
                dm[i]:mul(mom):add(1 - mom, g)
                g:copy(dm[i])
            end
        end
    end
}

-- function that implements Nesterov momentum:
-- this should be run *after* updateParameters()
optim.nesterov = argcheck{
    doc=[[
    <a name="optim.nesterov">
    #### optim.nesterov(@ARGP)
    @ARGT

    This function implements the Nesterov's accelerated gradient trick.  It
    creates a closure around a network which can later be called during the
    optimization.

    The closure should be run after `updateParameters()`. Its argument is
    the momentum parameter (must be a positive number).
    ]],
    {name = 'network',  type = 'nn.Container'},
    call = function(network)
        local ws = network:parameters()
        local t  = 1
        local m  = {[t] = {}, [t+1] = {}}
        for i, w in ipairs(ws) do
            m[t][i]   = w.new(w:size()):zero()
            m[t+1][i] = w.new(w:size()):zero()
        end
        return function(mom)
            assert(mom and type(mom) == 'number' and mom >= 0 and mom <= 1)
            if mom == 0 then return end
            for i, w in ipairs(ws) do
                m[t][i]:copy(m[t+1][i])
                m[t+1][i]:copy(w)   -- y_{t+1}
                w:mul(1-mom)        -- (1-g) * y_{t+1}
                w:add(mom, m[t][i]) -- g * y_t
            end
        end
    end
}

-- typically run after parameter update
optim.polyakAveraging = argcheck{
    doc=[[
    <a name="optim.polyakAveraging">
    #### optim.polyakAveraging(@ARGP)
    @ARGT

    This function implements the Nesterov's accelerated gradient trick.  It
    creates a closure around a network which can later be called during the
    optimization.

    The closure should be run after `updateParameters()`.

    The closure can be called with an optional boolean argument `usePolyak`.
    If `usePolyak` is set to `true`, then the weights of the model are replaced
    by their averaging.

    The closure returns the averaged weights.
    ]],
    {name = 'network', type = 'nn.Container'},
    call = function(network)
        local ws = network:parameters()
        local m  = {}
        for i, w in ipairs(ws) do
            m[i] = torch.Tensor(w:size()):typeAs(w):zero()
        end
        local t = 0
        return function(usePolyak)
            assert(usePolyak == nil or type(usePolyak) == 'boolean')
            for i, w in ipairs(ws) do
                if usePolyak then
                    w:copy(m[i])
                else
                    t = t + 1
                    m[i]:mul(1 - (1 / t)):add(1 / t, w)
                end
            end
            return m
        end
    end
}


return optim

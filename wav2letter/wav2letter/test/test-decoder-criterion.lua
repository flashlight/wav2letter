require 'torch'
require 'wav2letter'
local decoder = require 'wav2letter.runtime.decoder'

local datadir = assert(arg[1], 'datadir expected')
local eps = 1e-3

local function criterionJacobianTest1D(cri, input, target)
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   return (centraldiff_dfdx - dfdx):abs():max()
end

local function criterionJacobianTest1Dparams(cri, input, target, params, dfdx)
   cri:zeroGradParameters()
   cri:forward(input, target)
   cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local params_s = params:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,params:nElement() do
      -- f(xi + h)
      params_s[i] = params_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      params_s[i] = params_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      params_s[i] = params_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   return (centraldiff_dfdx - dfdx):abs():max()
end

--torch.manualSeed(1111)
torch.setdefaulttensortype('torch.FloatTensor')

decoder = decoder(
   paths.concat(datadir, 'letters-rep.lst'),
   paths.concat(datadir, 'dict-4g.lst'),
   paths.concat(datadir, 'librispeech-4-gram.bin'),
   'max', -- smearing
   -1 -- maxword
)

local dopt = {
   lmweight = 1,
   wordscore = 0,
   unkscore = -math.huge, -- NYI
   beamsize = 25, -- beware: small (~25) for GRAD TEST
   beamscore = 25,
   forceendsil = false,
   logadd = true -- false
}

local N = #decoder.letters+1
local criterion = nn.DecoderCriterion{
   decoder = decoder,
   dopt = dopt,
   N = N,
   scale =
      function(input, target)
         return math.sqrt(1/target:size(1))
      end
}
criterion.transitions:copy( torch.randn(N, N) )

local x = torch.rand(20, N)

-- DEBUG: beware to unknowns in target!!!
local target = " the cat sat "
local words = decoder.string2tensor(target)
target = target:gsub(' ', '|')

local y = torch.LongTensor(#target)
for i=1,#target do
   y[i] = criterion.decoder.letters[target:sub(i, i)] + 1
end

print(words)
criterion:setWordTarget(words)

-- criterion = nn.AutoSegCriterion(N, true, false, nil, nil, true) -- beware the scale
print('GRAD TEST', criterionJacobianTest1D(criterion, x, y))
print('GRAD TEST (PARAMS)', criterionJacobianTest1D(criterion, x, y, criterion.transitions, criterion.gtransitions))

local lr = 10 -- beware the scale
for i=1,10 do
   print("=======================================================================")
   print('iter', i, 'loss', criterion:forward(x, y))
   criterion:zeroGradParameters()
   criterion:backward(x, y)
   criterion:updateParameters(lr/1000)
   x:add(-lr, criterion.gradInput)
   local labels, llabels = criterion:labels()
   print('word (gld):', decoder.lettertensor2string(torch.add(y,-1)))
   print('word (dec):', decoder.tensor2string(decoder.removeneg(labels[1])))
   print("letter (fal):", decoder.lettertensor2string(criterion.__falpath))
   print("letter (dec)", decoder.lettertensor2string(llabels[1]))
end

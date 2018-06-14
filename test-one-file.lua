-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'io'
require 'torch'
require 'nn'
require 'wav2letter'

local sndfile = require 'sndfile'
local tnt = require 'torchnet'
local xlua = require 'xlua'
local serial = require 'wav2letter.runtime.serial'
local data = require 'wav2letter.runtime.data'
local transforms = require 'wav2letter.runtime.transforms'
local utils = require 'wav2letter.utils'
local decoder = require 'wav2letter.runtime.decoder'

torch.setdefaulttensortype('torch.FloatTensor')

local function cmdtestoptions(cmd)
   cmd:text()
   cmd:text('SpeechRec (c) Ronan Collobert 2015')
   cmd:text()
   cmd:text('Arguments:')
   cmd:argument('-model', 'the trained model!')
   cmd:argument('-wav', 'a sound file')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-gpu', 1, 'gpu device')
   cmd:option('-nolsm', false, 'remove lsm layer')
   cmd:option('-addlsm', false, 'add lsm layer')
   cmd:option('-letters', "", 'letters.lst')
   cmd:option('-lettersrep', "", 'letter-rep.lst')
   cmd:option('-words', "", 'words.lst')
   cmd:option('-maxword', -1, 'maximum number of words to use')
   cmd:option('-lm', "", 'lm.arpa.bin')
   cmd:option('-smearing', "none", 'none, max or logadd')
   cmd:option('-lmweight', 1, 'lm weight')
   cmd:option('-wordscore', 0, 'word insertion weight')
   cmd:option('-silweight', 0, 'weight for sil')
   cmd:option('-unkscore', -math.huge, 'unknown (word) insertion weight')
   cmd:option('-beamsize', 25000, 'max beam size')
   cmd:option('-beamscore', 40, 'beam score threshold')
   cmd:option('-forceendsil', false, 'force end sil')
   cmd:option('-logadd', false, 'use logadd instead of max')
   cmd:option('-nthread', 0, 'number of threads to use')
   cmd:option('-output', "", "an output file for the decoded results")
   cmd:text()
end

if #arg < 1 then
   error(string.format([[
usage:
   %s -model <options...>
]], arg[0]))
end

local reload = arg[1]
local opt = serial.parsecmdline{
   closure = cmdtestoptions,
   arg = arg,
   default = serial.loadmodel(reload).config.opt
}

if opt.gpu > 0 then
   require 'cutorch'
   require 'cunn'
   require 'cudnn'
   cutorch.setDevice(opt.gpu)
   cutorch.manualSeedAll(opt.seed)
end

--dictionaries
local dict = data.newdict{
   path = opt.letters
}

local dict39phn
if opt.target == "phn" then
   dict39phn = data.dictcollapsephones{dictionary=dict}
   if opt.dict39 then
      dict = dict39phn
   end
end

if opt.dictsil then
   data.dictadd{dictionary=dict, token='N', idx=assert(dict['|'])}
   data.dictadd{dictionary=dict, token='L', idx=assert(dict['|'])}
end

if opt.ctc or opt.garbage then
   data.dictadd{dictionary=dict, token="#"} -- blank
end

if opt.replabel > 0 then
   for i=1,opt.replabel do
      data.dictadd{dictionary=dict, token=string.format("%d", i)}
   end
end

print(string.format('| number of classes (network) = %d', #dict))

--reloading network
print(string.format('| reloading model <%s>', reload))
local model = serial.loadmodel{filename=reload, arch=true}
local network = model.arch.network
local transitions = model.arch.transitions
local config = model.config
local kw = model.config.kw
local dw = model.config.dw
assert(kw and dw, 'kw and dw could not be found in model archive')

if opt.nolsm then
   for i=network:size(),1,-1 do
      if torch.typename(network:get(i)) == 'nn.LogSoftMax' then
         print('! removing nn.LogSoftMax layer ' .. i)
         network:remove(i)
      end
   end
end
assert(not (opt.addlsm and opt.nolsm))
if opt.addlsm then
   if opt.gpu then
      network:insert(nn.LogSoftMax():cuda(), network:size())
   else
      network:add(nn.LogSoftMax())
   end
end
print(network)

-- make sure we do not apply aug on this
opt.aug = false
opt.shift = opt.shift or 0

local criterion

if opt.msc then
   criterion = nn.MultiStateFullConnectCriterion(#dict/opt.nstate, opt.nstate)
else
   criterion = (opt.ctc and nn.ConnectionistTemporalCriterion(#dict, nil)) or nn.Viterbi(#dict)
end
if not opt.ctc then
   criterion.transitions:copy(transitions)
end

if opt.shift > 0 then
   print(string.format("| using shift scheme (shift=%d dshift=%d)", opt.shift, opt.dshift))
   network = nn.ShiftNet(network, opt.shift)
end

local function tostring(tensor)
    local str = {}
    tensor:apply(
       function(idx)
          local letter = dict[idx]
          assert(letter)
          table.insert(str, letter)
       end
    )
    return table.concat(str)
 end

-- basic info
local fwav = sndfile.SndFile(opt.wav)
local fwavinfo = fwav:info()

if (fwavinfo.samplerate ~= opt.samplerate) then
    fwav:close()
    print(string.format('Invalid sample rate. The sample rate must be %sHz.', fwavinfo.samplerate))
    return
end

local wav = fwav:readFloat(fwavinfo.frames)
print(string.format('| number of frames: %d (%6.2fs) [samplerate: %d channels: %d]',
                    fwavinfo.frames, fwavinfo.frames/fwavinfo.samplerate, fwavinfo.samplerate, fwavinfo.channels))
fwav:close()

-- note that the current setup expect the whole file as a ByteTensor
local transformations = transforms.inputfromoptions(opt, kw, dw)
local netoutput = network:forward(transformations(wav))
local predictions = criterion:viterbi(netoutput)
predictions = utils.uniq(predictions)

print(string.format("\n| raw output (uniq): %s\n", tostring(predictions)))

decoder = decoder(
   opt.lettersrep,
   opt.words,
   opt.lm,
   opt.smearing,
   opt.maxword
)

local dopt = {
   lmweight = opt.lmweight,
   wordscore = opt.wordscore,
   unkscore = opt.unkscore,
   beamsize = opt.beamsize,
   beamscore = opt.beamscore,
   forceendsil = opt.forceendsil,
   logadd = opt.logadd,
   silweight = opt.silweight
}

local predictions, lpredictions = decoder(dopt, transitions, network.output)
predictions = decoder.removeunk(predictions)
predictions = decoder.tensor2string(predictions)

print(string.format("\n| decoded output: %s\n", predictions))

if #opt.output > 0 then
    file = io.open(opt.output, "w")
    file:write(predictions)
    file:close()
end

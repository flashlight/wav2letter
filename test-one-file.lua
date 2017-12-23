-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'wav2letter'

local sndfile = require 'sndfile'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1111)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('SpeechRec (c) Ronan Collobert 2015')
cmd:text()
cmd:text('Arguments:')
cmd:argument('-model', 'the trained model!')
cmd:argument('-wav', 'a sound file')
cmd:text()
cmd:text('Options:')
cmd:option('-datadir', string.format('%s/local/datasets/speech', os.getenv('HOME')), 'speech directory data')
cmd:text()

local testopt = cmd:parse(arg)

-- reload model
local f = torch.DiskFile(testopt.model):binary()
print('| best valid (test?) error was:', f:readObject())
local opt =  f:readObject()
local network = f:readObject()
local transitions = f:readObject()
f:close()

-- override
opt.datadir = testopt.datadir

-- only for tostring()
local config = paths.dofile(string.format('config/%s.lua', opt.config))
config = config(opt)
local transforms = (require 'wav2letter.runtime.transforms')(opt, config)

print(string.format('| number of classes (network) = %d', opt.nclass))

local criterion
if opt.msc then
   criterion = nn.MultiStateFullConnectCriterion(opt.nclass/opt.nstate, opt.nstate)
else
   criterion = nn.Viterbi(opt.nclass)
end
criterion.transitions:copy(transitions)

-- basic info
local fwav = sndfile.SndFile(testopt.wav)
local fwavinfo = fwav:info()
print(string.format('| number of frames: %d (%6.2fs) [samplerate: %d channels: %d]',
                    fwavinfo.frames, fwavinfo.frames/fwavinfo.samplerate, fwavinfo.samplerate, fwavinfo.channels))
fwav:close()

-- note that the current setup expect the whole file as a ByteTensor
local fwav = torch.DiskFile(testopt.wav):binary():quiet()
local wav = torch.ByteTensor(fwav:readByte(2^30))

local netoutput = network:forward(transforms.input(wav))
local output = transforms.remap(criterion:viterbi(netoutput))
print(string.format("| raw output (uniq): %s", config.tostring(output)))

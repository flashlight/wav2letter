-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'torch'
require 'nn'

local tnt = require 'torchnet'
local xlua = require 'xlua'
require 'wav2letter'
local serial = require 'wav2letter.runtime.serial'
local data = require 'wav2letter.runtime.data'

torch.setdefaulttensortype('torch.FloatTensor')

local function cmdtestoptions(cmd)
   cmd:text()
   cmd:text('SpeechRec (c) Ronan Collobert 2015')
   cmd:text()
   cmd:text('Arguments:')
   cmd:argument('-model', 'the trained model!')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-datadir', string.format('%s/local/datasets/speech', os.getenv('HOME')), 'speech directory data')
   cmd:option('-dictdir', string.format('%s/local/datasets/speech', os.getenv('HOME')), 'dictionary directory')
   cmd:option('-maxloadtest', -1, 'max number of testing examples')
   cmd:option('-show', false, 'show predictions')
   cmd:option('-save', false, 'save network predictions')
   cmd:option('-gpu', 1, 'gpu device')
   cmd:option('-nolsm', false, 'remove lsm layer')
   cmd:option('-addlsm', false, 'add lsm layer')
   cmd:option('-progress', false, 'display testing progress per epoch')
   cmd:option('-gfsai', false, 'override above paths to gfsai ones')
   cmd:option('-test', '', 'space-separated list of test data')
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

-- override paths?
if opt.gfsai then
   opt.datadir = '/mnt/vol/gfsai-east/ai-group/datasets/speech'
   opt.rundir = '/mnt/vol/gfsai-east/ai-group/users/' .. assert(os.getenv('USER'), 'unknown user') .. '/chronos'
end

if opt.gpu > 0 then
   require 'cutorch'
   require 'cunn'
   require 'cudnn'
   cutorch.setDevice(opt.gpu)
   cutorch.manualSeedAll(opt.seed)
end


--dictionaries
local dict = data.newdict{
   path = paths.concat(opt.dictdir, opt.dict)
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


local iterators = {}
for _, name in ipairs(data.namelist(opt.test)) do
   iterators[name] = data.newiterator{
   nthread = opt.nthread,
   closure =
      function()
         local data = require 'wav2letter.runtime.data'
         return data.newdataset{
            names = {name},
            opt = opt,
            dict = dict,
            kw = kw,
            dw = dw,
            maxload = opt.maxloadtest,
            words = 'wrd'
         }
      end
   }
end

local function createProgress(iterator)
   local N = iterator.execSingle
             and iterator:execSingle('size')
             or iterator:exec('size')
   local n = 0
   return function ()
      n = n + 1
      xlua.progress(n, N)
   end
end

local utils = require 'wav2letter.utils'
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

local words = {}
local wordsIdx = 0
local function string2wordtensor(str)
   local t = {}
   for word in str:gmatch('([^|]+)') do
      if not words[word] then
         words[word] = wordsIdx
         wordsIdx = wordsIdx + 1
      end
      table.insert(t, words[word])
   end
   return torch.LongTensor(t)
end

local function test(name, network, iterator)
   local fout
   local encodedName = string.gsub(name, '/', '-')
   if opt.save then
      -- outputs
      local path = paths.dirname(opt.model)
      fout = tnt.IndexedDatasetWriter{
         indexfilename = string.format("%s/output-%s.idx", path, encodedName),
         datafilename  = string.format("%s/output-%s.bin", path, encodedName),
         type = 'table',
      }
      -- transitions
      local f = torch.DiskFile(string.format("%s/transitions-%s.bin",
                                             path, encodedName), "w")
      f:binary()
      f:writeObject(transitions:float())
      f:close()
   end
   local engine = tnt.SGDEngine()
   local edit = tnt.EditDistanceMeter()
   local progress = opt.progress and createProgress(iterator)
   local wer = tnt.EditDistanceMeter()
   local iwer = tnt.EditDistanceMeter()
   function engine.hooks.onStart(state)
      edit:reset()
   end
   function engine.hooks.onForward(state)
      collectgarbage()
      local predictions = criterion:viterbi(state.network.output)
      predictions = utils.uniq(predictions)
      local targets = utils.uniq(state.sample.target)
      iwer:reset()
      local viterbi = tostring(predictions)
      local viterbiTensor = string2wordtensor(viterbi)
      local targetTensor = string2wordtensor(tostring(targets))
      iwer:add(viterbiTensor, targetTensor)
      wer:add(viterbiTensor, targetTensor)
      if opt.show then
         print(
            string.format("<%s>\n<%s>",
                          tostring(predictions),
                          tostring(targets))
         )
         print(
            string.format("[Sentence WER: %06.2f%%, dataset WER: %06.2f%%]",
                          iwer:value(), wer:value()))
      end
      if progress then
         progress()
      end
      if opt.save then
         fout:add{output=state.network.output,
                  spellings=targets,
                  words=state.sample.words,
                  predictions=tostring(predictions)}
      end
      edit:add(
         predictions,
         targets
      )
   end
   engine:test{
      network = network,
      iterator = iterator
   }
   if opt.save then
      fout:close()
   end
   return edit:value(), wer:value()
end

for name, iterator in pairs(iterators) do
   local ler, wer = test(name, network, iterator)
   print(string.format("| %s LER = %05.2f%%, WER = %05.2f%%",
                       name, ler, wer))
end

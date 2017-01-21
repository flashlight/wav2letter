require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

local tnt = require 'torchnet'
local xlua = require 'xlua'
require 'wav2letter'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1111)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('SpeechRec (c) Ronan Collobert 2015')
cmd:text()
cmd:text('Arguments:')
cmd:argument('-model', 'the trained model!')
cmd:text()
cmd:text('Options:')
cmd:option('-datadir', string.format('%s/local/datasets/speech', os.getenv('HOME')), 'speech directory data')
cmd:option('-maxloadtest', -1, 'max number of testing examples')
cmd:option('-show', false, 'show predictions')
cmd:option('-save', false, 'save network predictions')
cmd:option('-letters', '', 'letter dictionary to convert into strings (for -show and saving targets with -save)')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-nolsm', false, 'remove lsm layer')
cmd:option('-progress', false, 'display testing progress per epoch')
cmd:option('-gfsai', false, 'override above paths to gfsai ones')
cmd:text()

local testopt = cmd:parse(arg)

cutorch.setDevice(testopt.gpu)

-- reload model
local f = torch.DiskFile(testopt.model):binary()
local setup =  f:readObject()
local network
local transitions
local opt
if type(setup) == 'table' and setup.arg then
   local arch = f:readObject()
   opt = setup.opt
   network = arch.network
   transitions = arch.transitions
   print('| best valid (test?) error was:', setup.best)
else
   print("$ warning: loading old architecture")
   opt = f:readObject()
   network = f:readObject()
   transitions = f:readObject()
   print('| best valid (test?) error was:', type(setup) == 'table' and setup.editvalue or setup)
end
f:close()

-- override paths?
if testopt.gfsai then
   testopt.datadir = '/mnt/vol/gfsai-flash-east/ai-group/datasets/speech'
end

if testopt.nolsm then
   for i=network:size(),1,-1 do
      if torch.typename(network:get(i)) == 'nn.LogSoftMax' then
         print('! removing nn.LogSoftMax layer ' .. i)
         network:remove(i)
      end
   end
end
print(network)

-- make sure we do not apply aug on this
opt.aug = false
opt.shift = opt.shift or 0

opt.datadir = testopt.datadir
opt.maxloadtest = testopt.maxloadtest
local kw, dw = opt.kw, opt.dw

local config = paths.dofile(string.format('config/%s.lua', opt.config))
config = config(opt)
local transforms = paths.dofile('transforms.lua')(opt, config, false)

print(string.format('| number of classes (network) = %d', opt.nclass))

local criterion
if opt.msc then
   criterion = nn.MultiStateFullConnectCriterion(opt.nclass/opt.nstate, opt.nstate)
else
   criterion = (opt.ctc and nn.ConnectionistTemporalCriterion(opt.nclass, nil)) or nn.Viterbi(opt.nclass)
end
if not opt.ctc then
   criterion.transitions:copy(transitions)
end

if opt.shift > 0 then
   print(string.format("| using shift scheme (shift=%d dshift=%d)", opt.shift, opt.dshift))
   network = nn.ShiftNet(network, opt.shift)
end

local function filterbysize(sample)
   -- with opt.shift last one is smaller
   local input = opt.shift > 0 and sample.input[#sample.input] or sample.input
   local target = sample.target
   local isz = input:size(1)
   local tsz = target:size(1)
   local omaxisz = opt.maxisz
   local omaxtsz = opt.maxtsz
   if isz < kw+tsz*dw then
      return false
   end
   if omaxisz > 0 and isz > omaxisz then
      return false
   end
   if omaxtsz > 0 and tsz > omaxtsz then
      return false
   end
   return true
end

local iterators = {}
for name, test in pairs(config.testdatasets) do
   local dataset = tnt.TransformDataset{
      dataset = test(),
      transforms = {
         input = transforms.input,
         target = transforms.target
      }
   }
   if opt.shift > 0 then
      dataset = tnt.ShiftDataset{
         dataset = dataset,
         shift = opt.shift,
         dshift = opt.dshift,
         setshift = transforms.shift
      }
   end
   iterators[name] =
      tnt.DatasetIterator{
         dataset = dataset,
         filter = filterbysize
      }
end

local function createProgress(iterator)
   local N = iterator.execSingle and iterator:execSingle('size') or iterator:exec('size')
   local n = 0
   return function ()
      n = n + 1
      xlua.progress(n, N)
   end
end

local __letters = {}
if testopt.letters ~= '' then
   for letter in io.lines(testopt.letters) do
      table.insert(__letters, letter)
   end
end

local utils = require 'wav2letter.utils'
local function tostring(tensor)
   local str = {}
   tensor:apply(
      function(idx)
         local letter = __letters[idx]
         assert(letter)
         table.insert(str, letter)
      end
   )
   return table.concat(str)
end

local function test(name, network, iterator)
   local fout
   if testopt.save then
      -- outputs
      local path = paths.dirname(testopt.model)
      fout = tnt.IndexedDatasetWriter{
         indexfilename = string.format("%s/output-%s.idx", path, name),
         datafilename  = string.format("%s/output-%s.bin", path, name),
         type = 'table',
      }
      -- transitions
      local f = torch.DiskFile(string.format("%s/transitions-%s.bin", path, name), "w")
      f:binary()
      f:writeObject(transitions:float())
      f:close()
   end
   local engine = tnt.SGDEngine()
   local edit = tnt.EditDistanceMeter()
   local progress = testopt.progress and createProgress(iterator)
   function engine.hooks.onStart(state)
      edit:reset()
   end
   function engine.hooks.onForward(state)
      collectgarbage()
      local predictions = criterion:viterbi(state.network.output)
      predictions = utils.uniq(predictions)
      local targets = utils.uniq(state.sample.target)
      if testopt.show then
         print(
            string.format("<%s> || <%s>",
                          tostring(predictions),
                          tostring(targets))
         )
      end
      if progress then
         progress()
      end
      if testopt.save then
         fout:add{output=state.network.output, spellings=targets, words=state.sample.words}
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
   if testopt.save then
      fout:close()
   end
   return edit:value()
end

for name, iterator in pairs(iterators) do
   print(string.format("| %s error = %05.2f%%", name, test(name, network, iterator)))
end

require 'torch'
require 'nn'
require 'fbnn'
require 'fb.debugger'

local tnt = require 'fbtorchnet'
local xlua = require 'xlua'
local threads = require 'threads'
local logtext = require 'torchnet.log.view.text'

require 'wav2letter'

torch.setdefaulttensortype('torch.FloatTensor')

local netutils = paths.dofile('netutils.lua')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('SpeechRec (c) Ronan Collobert 2015')
cmd:text()
cmd:text('Options:')
cmd:option('-datadir', string.format('%s/local/datasets/speech', os.getenv('HOME')), 'speech directory data')
cmd:option('-rundir', string.format('%s/local/experiments/speech', os.getenv('HOME')), 'experiment root directory')
cmd:option('-archdir', string.format('%s/local/arch/speech', os.getenv('HOME')), 'arch root directory')
cmd:option('-gfsai', false, 'override above paths to gfsai ones')
cmd:option('-hashdir', false, 'hash experiment directory name')
cmd:option('-outputexample', false, 'write out examples into current directory')
cmd:option('-seed', 1111, 'Manually set RNG seed')
cmd:option('-progress', false, 'display training progress per epoch')
cmd:option('-arch', 'default', 'network architecture')
cmd:option('-archgen', '', 'network architecture generator string')
cmd:option('-batchsize', 1, 'batchsize')
cmd:option('-config', 'timit', 'config setup (timit, switchboard)')
cmd:option('-linseg', 0, 'number of linear segmentation iter, if not using -seg')
cmd:option('-linsegznet', false, 'use fake zero-network with linseg')
cmd:option('-linlr', -1, 'linear segmentation learning rate (if < 0, use lr)')
cmd:option('-linlrcrit', -1, 'linear segmentation learning rate (if < 0, use lrcrit)')
cmd:option('-absclamp', 0, 'if > 0, clamp gradient to -value..value')
cmd:option('-scaleclamp', 0, 'if > 0, clamp gradient to -(scale*|w|+value)..(scale*|w|+value) (value provided by -absclamp)')
cmd:option('-normclamp', 0, 'if > 0, clamp gradient to provided norm')
cmd:option('-falseg', 0, 'number of force aligned segmentation iter')
cmd:option('-fallr', -1, 'force aligned segmentation learning rate (if < 0, use lr)')
cmd:option('-iter', 1000000,   'number of iterations')
cmd:option('-itersz', -1, 'iteration size')
cmd:option('-lr', 1, 'learning rate')
cmd:option('-layerlr', false, 'use learning rate per layer (divide by number of inputs)')
cmd:option('-lrcrit', 0, 'criterion learning rate')
cmd:option('-lblwin', 21, 'number of frames to estimate the center label (seg=true only)')
cmd:option('-gpu', 0, 'use gpu instead of cpu (indicate device > 0)')
cmd:option('-nthread', 1, 'specify number of threads for data parallelization')
cmd:option('-posmax', false, 'use max instead of logadd (pos)')
cmd:option('-negmax', false, 'use max instead of logadd (neg)')
cmd:option('-inormmax', false, 'input norm is max instead of std')
cmd:option('-inormloc', false, 'input norm is local instead global')
cmd:option('-inkw', 8000, 'local input norm kw')
cmd:option('-indw', 2666, 'local input norm dw')
cmd:option('-innt', 0.01, 'local input noise threshold')
cmd:option('-onorm', 'none', 'output norm (none, input or target)')
cmd:option('-maxload', -1, 'max number of training examples (random sub-selection)')
cmd:option('-maxloadseed', 1111, 'seed for random sub-selection of training examples')
cmd:option('-maxloadvalid', -1, 'max number of valid examples (linear sub-selection)')
cmd:option('-maxloadtest', -1, 'max number of testing examples (linear sub-selection)')
cmd:option('-momentum', -1, 'provide momentum')
cmd:option('-mtcrit', false, 'use multi-threaded criterion')
cmd:option('-nstate', 1, 'number of states per label (autoseg only)')
cmd:option('-msc', false, 'use multi state criterion instead of fcc')
cmd:option('-ctc', false, 'use ctc criterion for training')
cmd:option('-garbage', false, 'add a garbage between each target label')
cmd:option('-maxisz', -1, 'max input size allowed during training')
cmd:option('-maxtsz', -1, 'max target size allowed during training')
cmd:option('-mintsz', -1, 'min target size allowed during training')
cmd:option('-reload', '', 'reload a particular model')
cmd:option('-reloadarg', false, 'reload argument string')
cmd:option('-continue', '', 'continue a particular model')
cmd:option('-force', false, 'force overwriting experiment')
cmd:option('-noresample', false, 'do not resample training data')
cmd:option('-terrsr', 1, 'train err sample rate (default: each example; 0 is skip)')
cmd:option('-psr', 0, 'perf (statistics) print sample rate (default: only at the end of epochs)')
cmd:option('-replabel', -1, 'replace up to replabel reptitions by additional classes')
cmd:option('-lsm', false, 'add LogSoftMax layer')
cmd:option('-tag', '', 'tag this experiment with a particular name (e.g. "hypothesis1")')

cmd:text()
cmd:text('MFCC Options:')
cmd:option('-mfcc', false, 'use standard htk mfcc features as input')
cmd:option('-pow', false, 'use standard power spectrum as input')
cmd:option('-mfcc_mel_floor', 0.0, 'specify optional mfcc mel floor')
cmd:option('-mfcc_coefs', 13, 'number of mfcc coefficients')
cmd:option('-mfsc', false, 'use standard mfsc features as input')

cmd:text()
cmd:text('Data Augmentation Options:')
cmd:option('-aug', false, 'Enable data augmentations')
cmd:option('-bending', -1, 'Enable pitch bending with given probability')
cmd:option('-caug', false, 'enable flanger')
cmd:option('-eaug', false, 'enable chorus and echos')
cmd:option('-noise', -1, 'Enable addition of white/brown noise with given probability')
cmd:option('-vaug', false, 'enable companding (may clip!)')
cmd:option('-saug', -1, 'variance of input speed transformation')
cmd:option('-saugp', 1.0, 'probability with which input speech transformation is applied')

cmd:text()
cmd:text('Timit-Only Options:')
cmd:option('-seg', false, 'segmentation is given or not')
cmd:option('-dict39', false, 'dictionary with 39 phonemes mode (training -- always for testing)')
cmd:option('-discardq', false, 'map q to silence')
cmd:text()

cmd:text()
cmd:text('Switchboard-Only Options:')
cmd:option('-swbst', 'trans', 'switchboard transcription subtype (trans or word)')
cmd:text()

cmd:text()
cmd:text('LibriSpeech-Only Options:')
cmd:option('-lsc100', false, 'use clean 100h of speech')
cmd:option('-lsc360', false, 'use extra clean 360h of speech')
cmd:option('-lso500', false, 'use extra other 500h of speech')
cmd:option('-lfisher', false, 'use fisher')
cmd:option('-lswb', false, 'use switchboard')
cmd:option('-lmessenger', false, 'use messenger')
cmd:option('-l8khz', false, 'downsample data to 8khz (if necessary)')
cmd:option('-surround', false, 'surround target with spaces')
cmd:option('-words', 0, 'use words (indicate how many) instead of letters')
cmd:text()

cmd:text('WSJ-Only Options:')
cmd:option('-si84', false, 'limit to si84 dataset for training')
cmd:text()

cmd:text('Input shifting Options:')
cmd:option('-shift', 0, 'number of shifts')
cmd:option('-dshift', 0, '# of frames to shift')
cmd:option('-gpushift', false, 'use one GPU per shift')
cmd:text()

local opt = cmd:parse(arg)
local dbg = {} --debugging information (saved first)

if opt.gpu > 0 then
   cutorch.setDevice(opt.gpu)
end

local function mkdir(path)
   os.execute(string.format('mkdir -p "%s"', path))
end

if opt.continue ~= '' then
   opt.reload = opt.continue
   opt.reloadarg = true
end

if opt.reload ~= '' and opt.reloadarg then
   print(string.format('| Reloading options <%s>', opt.reload))
   local f = torch.DiskFile(opt.reload):binary()
   local setup = f:readObject()
   if setup.opt.gpu > 0 then
      require 'cunn'
      require 'fbcunn'
      require 'cudnn'
   end
   local reloadArg = setup.arg
   if opt.continue ~= '' then
      print('| Adding current options')
      for i = 1, #arg do
         print(string.format("  | %s", arg[i]))
         reloadArg[#reloadArg+1] = arg[i]
      end
   end
   arg = reloadArg
   local reload = opt.reload
   opt = cmd:parse(arg)
   opt.reload = reload -- make sure we reload the model below
end

-- override paths?
if opt.gfsai then
   opt.datadir = '/mnt/vol/gfsai-flash-east/ai-group/datasets/speech'
   opt.rundir = '/mnt/vol/gfsai-east/ai-group/teams/wav2letter/experiments'
   opt.archdir = '/mnt/vol/gfsai-east/ai-group/teams/wav2letter/arch'
end


dbg.name = cmd:string(
   'exp',
   opt,
   {
      force=true, gfsai=true,
      datadir=true, rundir=true, archdir=true,
      iter=true, gpu=true, reload=true, progress=true,
      continue=true
   }
)
print("| ExpName: " .. dbg.name)

local path = opt.reload == '' and opt.rundir or paths.dirname(opt.reload)
if opt.hashdir then
   -- hash based on experiment name
   path = paths.concat(path, tnt.utils.sys.md5(dbg.name))
else
   path = paths.concat(path, dbg.name)
end

-- check if experiment exists
if not opt.force then
   local f = io.open(path .. '/log')
   if f then
      f:close()
      error(string.format('experiment <%s> already exists! use -force to overwrite', path))
   end
end

opt.path = path
print(string.format("| experiment path: %s", path))
mkdir(path)

-- collect debug info
dbg.username = os.getenv('USER')
dbg.hostname = os.getenv('HOSTNAME')
dbg.timestamp = os.time()

-- default lr
opt.linlr = (opt.linlr < 0) and opt.lr or opt.linlr
opt.fallr = (opt.fallr < 0) and opt.lr or opt.fallr
opt.linlrcrit = (opt.linlrcrit < 0) and opt.lrcrit or opt.linlrcrit

torch.manualSeed(opt.seed)
if opt.gpu > 0 then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   cutorch.manualSeedAll(opt.seed)
end

local config = paths.dofile(string.format('config/%s.lua', opt.config))
config = config(opt)

opt.dataspecs = config.specs
opt.nchannel = opt.dataspecs.nchannel
opt.samplerate = opt.dataspecs.samplerate
opt.nclass = config.specs.nclass
if opt.replabel > 0 then
   opt.nclass = opt.nclass + opt.replabel
end

if opt.garbage then
   assert(opt.nstate == 1, 'cannot have garbage and nstate set together')
   opt.nclass = opt.nclass + 1
else
   opt.nclass = opt.nclass*opt.nstate
end

print(string.format('| number of classes (network) = %d', opt.nclass))

-- neural network and training criterion
-- we make sure we save the network specs
-- optional argument archgen generates the network file on the fly
if opt.archgen ~= '' then
   local arch_s = paths.dofile('arch_gen/conv_gen.lua')(opt.archgen)
   local arch_f = io.open(paths.concat(opt.archdir, opt.archgen), "wb")
   arch_f:write(arch_s)
   arch_f:close()
   opt.arch = opt.archgen
end
opt.netspecs = netutils.readspecs(paths.concat(opt.archdir, opt.arch))
local network, kw, dw = netutils.create(opt.netspecs, opt.gpu, opt.dataspecs.nchannel, opt.nclass, opt.lsm)
local zeronet = nn.ZeroNet(kw, dw, opt.nclass)
local netcopy = network:clone() -- pristine stateless copy
opt.kw = kw
opt.dw = dw
local scale
if opt.onorm == 'input' then
   function scale(input, target)
      return 1/input:size(1)
   end
elseif opt.onorm == 'target' then
   function scale(input, target)
      return 1/target:size(1)
   end
elseif opt.onorm ~= 'none' then
   error('invalid onorm option')
end
print(string.format('| neural network number of parameters: %d', netutils.size(network)))

local function initCriterion(class, ...)
   if opt.batchsize > 1 and class == 'AutoSegCriterion' then
      return nn.BatchAutoSegCriterionC(opt.batchsize, ...)
   elseif opt.batchsize > 1 and opt.mtcrit then
      return nn.MultiThreadedBatchCriterion(opt.batchsize, {'transitions'}, class, ...)
   elseif opt.batchsize > 1 then
      return nn.BatchCriterion(opt.batchsize, {'transitions'}, class, ...)
   else
      return nn[class](...)
   end
end

local fllcriterion
local asgcriterion
local ctccriterion = initCriterion('ConnectionistTemporalCriterion', opt.nclass, scale)
local msccriterion = initCriterion('MultiStateFullConnectCriterion', opt.nclass/opt.nstate, opt.nstate, opt.posmax, scale)
local lincriterion = initCriterion('LinearSegCriterion', opt.nclass, opt.negmax, scale, opt.linlrcrit == 0)
local falcriterion = initCriterion('CrossEntropyForceAlignCriterion', opt.nclass, opt.posmax, scale)
local viterbi      = initCriterion('Viterbi', opt.nclass, scale)

if opt.garbage then
   fllcriterion = initCriterion('FullConnectGarbageCriterion', opt.nclass-1, opt.posmax, scale)
   asgcriterion = initCriterion('AutoSegCriterion', opt.nclass-1, opt.posmax, opt.negmax, scale, 'garbage')
else
   fllcriterion = initCriterion('FullConnectCriterionC', opt.nclass, opt.posmax, scale)
   asgcriterion = initCriterion('AutoSegCriterion', opt.nclass, opt.posmax, opt.negmax, scale, opt.msc and opt.nstate or nil)
end

lincriterion:share(asgcriterion, 'transitions') -- beware (asg...)
falcriterion:share(asgcriterion, 'transitions')
fllcriterion:share(asgcriterion, 'transitions')
msccriterion:share(asgcriterion, 'transitions')
viterbi:share(asgcriterion, 'transitions')

local evlcriterion = (opt.ctc and ctccriterion) or (opt.msc and msccriterion or viterbi)
-- clone is important (otherwise forward/backward not in a row
-- because we evaluate right after the forward and before the backward)
evlcriterion = evlcriterion:clone():share(asgcriterion, 'transitions')

if opt.reload ~= '' then
   print(string.format('| reloading model <%s>', opt.reload))
   local f = torch.DiskFile(opt.reload):binary()
   f:readObject() -- setup
   local arch = f:readObject()
   network = arch.network
   asgcriterion.transitions:copy(arch.transitions)
   arch = nil
   collectgarbage()
end

if opt.layerlr then
   network = netutils.layerlr(network, opt.lr)
end

if opt.momentum > 0 then
   network = netutils.momentum(network, opt.momentum)
end

local function applyClamp() end
local wavoptim = require 'wav2letter.optim'
if opt.scaleclamp > 0 then
   local apply = wavoptim.weightedGradientClamp(network, asgcriterion)
   function applyClamp()
      apply(opt.absclamp, opt.scaleclamp)
   end
elseif opt.absclamp > 0 then
   local apply = wavoptim.absGradientClamp(network, asgcriterion)
   function applyClamp()
      apply(opt.absclamp)
   end
elseif opt.normclamp > 0 then
   local apply = wavoptim.normGradientClamp(network, asgcriterion)
   function applyClamp()
      apply(opt.normclamp)
   end
end

local function makeParallel(network, size)
   local _network = nn.DataParallelTableTable(true, true):threads(function()
        require 'cudnn'
        require 'fbcunn'
        cudnn.fastest = true
     end)
   local gpus = {}
   for i = 1, size do
      gpus[i] = opt.gpu - 1 + i
   end
   _network:add(network, gpus)
   return _network
end

assert(not(opt.batchsize > 1 and opt.shift > 0), 'Cannot allow both shifting and batching')

if opt.batchsize > 1 then
   network = makeParallel(network, opt.batchsize)
end

if opt.shift > 0 then
   if opt.gpushift then
      network = makeParallel(network, opt.shift)
   else
      network = nn.MapTable(network, {'weight', 'bias'})
      network:resize(opt.shift)
   end
   network = nn.ShiftNet(network, opt.shift, opt.gpushift)
end

local transformsTrain = paths.dofile('transforms.lua')(opt, config, opt.aug)
local transformsTest = paths.dofile('transforms.lua')(opt, config, false)

local function filterbysize(sample)
   -- with opt.shift last one is smaller
   local input = opt.shift > 0 and sample.input[#sample.input] or sample.input
   local target = sample.target
   local isz = opt.batchsize > 1 and input[1]:size(1) or input:size(1)
   local tsz = opt.batchsize > 1 and target[1]:size(1) or target:size(1)
   if isz < kw+tsz*dw then
      return false
   end
   if opt.batchsize > 1 then
      for i = 2, #target do
         local iszI = input[i]:size(1)
         local tszI = target[i]:size(1)
         if iszI < kw+tszI*dw then
            return false
         end
         tsz = math.max(tsz, tszI)
         isz = math.max(isz, iszI)
      end
   end
   if opt.maxisz > 0 and isz > opt.maxisz then
      return false
   end
   if tsz < opt.mintsz then
      print("warning ca va peter -- filtered out")
      return false
   end
   if opt.maxtsz > 0 and tsz > opt.maxtsz then
      return false
   end
   return true
end

local resampleperm = torch.LongTensor()
local resample =
   threads.safe(
      function(size)
         if size then
            if resampleperm:nDimension() == 0 then
               print('# resampling: init with', size)
               resampleperm:randperm(size)
            else
               assert(
                  resampleperm:nDimension() == 1 and
                     resampleperm:size(1) == size
               )
               print('# resampling: skip init')
            end
         else
            print('# resampling with', resampleperm:size(1))
            resampleperm:randperm(resampleperm:size(1))
         end
      end
   )

local trainsort = torch.LongTensor()
local setTrainSort = threads.safe(
   function(dataset)
      if trainsort:dim() == 0 then
         trainsort:resize(dataset:size()):zero()
      end
   end
)

local function buildPerm(dataset)
   setTrainSort(dataset)
   return function(idx, size)
      if trainsort[idx] == 0 then
         trainsort[idx] = dataset:get(idx).input:size(1)
         return idx
      else
         return trainsort[idx]
      end
   end
end

local trainiterator

if opt.nthread == 0 then
   local traindataset = tnt.TransformDataset{
      dataset = config.traindataset(),
      transforms = {
         input = transformsTrain.input,
         target = transformsTrain.target
      }
   }
   assert(opt.itersz <= traindataset:size()) -- DEBUG: fixit
   resample(traindataset:size())
   traindataset = tnt.ResampleDataset{
      dataset = traindataset,
      sampler =
         function(self, idx)
            return resampleperm[idx]
         end,
      size = opt.itersz
   }
   if opt.shift > 0 then
      traindataset = tnt.ShiftDataset{
         dataset = traindataset,
         shift = opt.shift,
         dshift = opt.dshift,
         setshift = transformsTrain.shift
      }
   end
   if opt.batchsize > 1 then
      traindataset = tnt.BatchDataset{
         dataset = traindataset,
         batchsize = opt.batchsize,
         merge = function(sample) return sample end,
         perm = buildPerm(traindataset),
      }
   end
   trainiterator = tnt.DatasetIterator{
      dataset = traindataset,
      filter = filterbysize
   }
else
   trainiterator = tnt.ParallelDatasetIterator{
      closure = function(idx)
         local config = paths.dofile(string.format('config/%s.lua', opt.config))
         config = config(opt)
         local transformsTrain = paths.dofile('transforms.lua')(opt, config, opt.aug, idx)
         local tnt = require 'fbtorchnet'
         require 'wav2letter' -- ShiftDataset
         local traindataset = tnt.TransformDataset{
            dataset = config.traindataset(),
            transforms = {
               input = transformsTrain.input,
               target = transformsTrain.target
            }
         }
         assert(opt.itersz <= traindataset:size()) -- DEBUG: fixit
         resample(traindataset:size())
         traindataset = tnt.ResampleDataset{
            dataset = traindataset,
            sampler =
               function(self, idx)
                  return resampleperm[idx]
               end,
            size = opt.itersz
         }
         if opt.shift > 0 then
            traindataset = tnt.ShiftDataset{
               dataset = traindataset,
               shift = opt.shift,
               dshift = opt.dshift,
               setshift = transformsTrain.shift
            }
         end
         if opt.batchsize > 1 then
            traindataset = tnt.BatchDataset{
               dataset = traindataset,
               batchsize = opt.batchsize,
               merge = function(sample) return sample end,
               perm = buildPerm(traindataset),
            }
         end
         return traindataset
      end,
      nthread=opt.nthread,
      filter = filterbysize
   }
end
local trainsize = trainiterator.execSingle and trainiterator:execSingle('size') or trainiterator:exec('size')

local validiterators = {}
for name, valid in pairs(config.validdatasets) do
   local dataset = tnt.TransformDataset{
            dataset = valid(),
            transforms = {
               input = transformsTest.input,
               target = transformsTest.target
            }
         }
   if opt.shift > 0 then
      dataset = tnt.ShiftDataset{
         dataset = dataset,
         shift = opt.shift,
         dshift = opt.dshift,
         setshift = transformsTest.shift
      }
   end
   if opt.batchsize > 1 then
      dataset = tnt.BatchDataset{
         dataset = dataset,
         batchsize = opt.batchsize,
         merge = function(sample) return sample end,
      }
   end
   validiterators[name] =
      tnt.DatasetIterator{
         dataset = dataset,
         filter = filterbysize
      }
end

local testiterators = {}
for name, test in pairs(config.testdatasets) do
   local dataset = tnt.TransformDataset{
            dataset = test(),
            transforms = {
               input = transformsTest.input,
               target = transformsTest.target
            }
         }
   if opt.shift > 0 then
      dataset = tnt.ShiftDataset{
         dataset = dataset,
         shift = opt.shift,
         dshift = opt.dshift,
         setshift = transformsTest.shift
      }
   end
   if opt.batchsize > 1 then
      dataset = tnt.BatchDataset{
         dataset = dataset,
         batchsize = opt.batchsize,
         merge = function(sample) return sample end,
      }
   end
   testiterators[name] =
      tnt.DatasetIterator{
         dataset = dataset,
         filter = filterbysize
      }
end

----------------------------------------------------------------------
-- Performance meters

local meters = {}

meters.runtime = tnt.TimeMeter()
meters.timer = tnt.TimeMeter{unit = true}
meters.sampletimer = tnt.TimeMeter{unit = true}
meters.networktimer = tnt.TimeMeter{unit = true}
meters.criteriontimer = tnt.TimeMeter{unit = true}
meters.loss = tnt.AverageValueMeter{}
if opt.seg then -- frame error rate
   meters.trainframeerr = tnt.FrameErrorMeter{}
end
meters.trainedit = tnt.EditDistanceMeter()

meters.validedit = {}
for name, valid in pairs(config.validdatasets) do
   meters.validedit[name] = tnt.EditDistanceMeter()
end

meters.testedit = {}
for name, test in pairs(config.testdatasets) do
   meters.testedit[name] = tnt.EditDistanceMeter()
end

meters.stats = tnt.SpeechStatMeter()
meters.bdev = tnt.AverageValueMeter{}

local logfile = torch.DiskFile(
   string.format('%s/perf', opt.path),
   (opt.continue == '') and "w" or "rw"
)

local function status(state)
   local ERR = opt.words > 0 and 'WER' or 'LER'
   local status = {
      string.format("epoch %4.2f", state.epoch),
      string.format("lr %4.6f", state.lr),
      string.format("lrcriterion %4.6f", state.lrcriterion),
      string.format("runtime(h) %4.2f", meters.runtime:value()/3600),
      string.format("ms(bch) %4d", meters.timer:value()*1000),
      string.format("ms(smp) %4d", meters.sampletimer:value()*1000),
      string.format("ms(net) %4d", meters.networktimer:value()*1000),
      string.format("ms(crt) %4d", meters.criteriontimer:value()*1000),
      string.format("loss %10.5f", meters.loss:value()),
   }
   if opt.seg then
      table.insert(
         status,
         string.format("train ferr %5.2f", meters.trainframeerr:value())
      )
   end
   table.insert(
      status,
      string.format('train %s %5.2f', ERR, meters.trainedit:value())
   )
   for name, meter in pairs(meters.validedit) do
      table.insert(
         status,
         string.format('%s %s %5.2f', name, ERR, meter:value())
      )
   end
   for name, meter in pairs(meters.testedit) do
      table.insert(
         status,
         string.format('%s %s %5.2f', name, ERR, meter:value())
      )
   end
   local stats = meters.stats:value()
   table.insert(
      status,
      string.format(
         "%03d aisz %03d atsz %03d mtsz %7.2fh",
         stats['isz']/stats['n'],
         stats['tsz']/stats['n'],
         stats['maxtsz'],
         stats['isz']/opt.dataspecs.samplerate/3600)
   )
   if opt.batchsize > 1 then
      table.insert(
         status,
         string.format("bdev %7.2f", meters.bdev:value()*100)
      )
   end

   -- print and log
   status = table.concat(status, " | ")
   print(status)
   logfile:seekEnd()
   logfile:writeString(status)
   logfile:writeString("\n")
   logfile:synchronize()
end

-- best perf so far on valid datasets
local minerrs = {}
for name, valid in pairs(config.validdatasets) do
   minerrs[name] = math.huge
end

local function save(name, network, best)
   cutorch.synchronizeAll()
   local f = torch.DiskFile(string.format('%s/model-%s.bin', opt.path, name), 'w')
   f:binary()
   f:writeObject{
      best = best,
      opt = opt,
      arg = arg
   }
   f:writeObject{
      network = netutils.copy(
         netcopy,
         (opt.shift > 0) and network.network or network
      ),
      transitions = asgcriterion.transitions
   }
   f:close()
end

local function savebestmodels()
   -- save last model
   save("last", network)

   -- save if better than ever for one valid
   local best = {}
   for name, validedit in pairs(meters.validedit) do
      local value = validedit:value()
      if value < minerrs[name] then
         best[name] = value
         minerrs[name] = value
         save(name, network, value)
      end
   end

   return best
end

----------------------------------------------------------------------


local function createProgress(iterator)
   local N = iterator.execSingle and iterator:execSingle('size') or iterator:exec('size')
   local n = 0
   return function ()
      n = n + 1
      xlua.progress(n, N)
   end
end

local function map(closure, a)
   if type(a) == 'table' then
      for k,v in pairs(a) do
         map(closure, a[k])
      end
   else
      closure(a)
   end
end

local function map2(closure, a, b)
   if type(a) == 'table' then
      for k,v in pairs(a) do
         map2(closure, a[k], b[k])
      end
   else
      closure(a, b)
   end
end

local function collectBdev(sample)
   local minv = sample.input[1]:size(1)
   local maxv = minv
   local meanv = 0
   for i = 2, #sample.input do
      minv = math.min(minv, sample.input[i]:size(1))
      maxv = math.max(maxv, sample.input[i]:size(1))
      meanv = meanv + sample.input[i]:size(1)
   end
   return (maxv - minv) / maxv
end

local function evalOutput(edit, output, target, transforms)
   local function evl(o, t)
      if o:numel() > 0 then
         edit:add(transforms.remap(o), transforms.remap(t))
      end
   end
   map2(evl, evlcriterion:viterbi(output), target)
end

local function test(network, criterion, iterator, edit)
   local progress = opt.progress and createProgress(iterator)
   local engine = tnt.SGDEngine()
   function engine.hooks.onStart()
      edit:reset()
   end
   function engine.hooks.onForward(state)
      if progress then
         progress()
      end
      collectgarbage()
      evalOutput(edit, state.network.output, state.sample.target, transformsTest)
   end
   engine:test{
      network = network,
      iterator = iterator
   }
   if progress then
      print()
   end
   return edit:value()
end

meters.runtime:reset()
local function train(network, criterion, iterator, params, opid)
   local progress
   local engine = tnt.SGDEngine()

   function engine.hooks.onStart(state)
      meters.bdev:reset()
      meters.loss:reset()
      meters.trainedit:reset()
   end

   function engine.hooks.onStartEpoch(state)
      if not opt.noresample then
         resample()
      end
      if trainframeerr then
         trainframeerr:reset()
      end
      progress = opt.progress and createProgress(iterator)
      meters.stats:reset()
      meters.timer:reset()
      meters.sampletimer:resume()
      meters.sampletimer:reset()
      meters.networktimer:stop()
      meters.networktimer:reset()
      meters.criteriontimer:stop()
      meters.criteriontimer:reset()
      meters.timer:resume()
   end

   function engine.hooks.onSample(state)
      if progress then
         progress()
      end
      meters.sampletimer:stop()
      meters.networktimer:resume()
   end

   function engine.hooks.onForward(state)
      if opt.batchsize > 1 then meters.bdev:add(collectBdev(state.sample)) end
      meters.networktimer:stop()
      meters.criteriontimer:resume()
      if state.t % opt.terrsr == 0 then
         evalOutput(meters.trainedit, state.network.output, state.sample.target, transformsTrain)
      end
      if trainframeerr then
         evalOutput(meters.trainframeerr, state.network.output, state.sample.target, transformsTrain)
      end
   end

   function engine.hooks.onBackwardCriterion(state)
      meters.criteriontimer:stop()
      meters.networktimer:resume()
   end

   function engine.hooks.onBackward(state)
      applyClamp()
      meters.networktimer:stop()
   end

   function engine.hooks.onUpdate(state)
      map(function(out) if out then meters.loss:add(out) end end, state.criterion.output)
      map2(function(i, t) if i then meters.stats:add(i, t) end end, opt.shift > 0 and state.sample.input[1] or state.sample.input, state.sample.target)
      meters.timer:incUnit()
      meters.sampletimer:incUnit()
      meters.networktimer:incUnit()
      meters.criteriontimer:incUnit()
      meters.sampletimer:resume()
      if state.t % opt.psr == 0 and state.t % trainsize ~= 0 then
         if progress then
            print()
         end

         -- print status
         status(state)

         -- save last and best models
         savebestmodels()

         -- Reset average value meters (so that we average over opt.psr steps)
         meters.bdev:reset()
         meters.loss:reset()
         meters.trainedit:reset()
      end
   end

   function engine.hooks.onEndEpoch(state)
      meters.timer:stop()
      if progress then
         print()
      end

      -- valid
      for name, validiterator in pairs(validiterators) do
         test(network, criterion, validiterator, meters.validedit[name])
      end

      -- test
      for name, testiterator in pairs(testiterators) do
         test(network, criterion, testiterator, meters.testedit[name])
      end

      -- print status
      status(state)

      -- save last and best models
      savebestmodels()

      -- reset meters for next readings
      meters.bdev:reset()
      meters.loss:reset()
      meters.trainedit:reset()

      -- sort indices (builds batches of similar length)
      if opt.batchsize > 1 and state.epoch == 1 then
         local _,itrainsort = torch.sort(trainsort)
         trainsort:copy(itrainsort)
      end
   end

   engine:train{
      network = network,
      criterion = criterion,
      iterator = iterator,
      lr = params.lr,
      lrcriterion = params.lrcriterion,
      maxepoch = params.maxepoch
   }
end

if not opt.seg and opt.linseg > 0 then
   train(
      opt.linsegznet and zeronet or network,
      lincriterion,
      trainiterator,
      {lr=opt.linlr, lrcriterion=opt.linlrcrit, maxepoch=opt.linseg},
      1
   )
end

if opt.falseg > 0 then
   train(
      network,
      falcriterion,
      trainiterator,
      {lr=opt.fallr, maxepoch=opt.falseg},
      2
   )
end

train(
   network,
   (opt.ctc and ctccriterion) or (opt.seg and fllcriterion or asgcriterion),
   trainiterator,
   {lr=opt.lr, lrcriterion=opt.lrcrit, maxepoch=opt.iter},
   3
)

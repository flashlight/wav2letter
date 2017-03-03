require 'torch'
require 'nn'

local tnt = require 'torchnet'
local threads = require 'threads'
local data = paths.dofile('data.lua')
local log = paths.dofile('log.lua')

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
cmd:option('-mpi', false, 'use mpi parallelization')
cmd:option('-outputexample', false, 'write out examples into current directory')
cmd:option('-seed', 1111, 'Manually set RNG seed')
cmd:option('-progress', false, 'display training progress per epoch')
cmd:option('-arch', 'default', 'network architecture')
cmd:option('-archgen', '', 'network architecture generator string')
cmd:option('-batchsize', 0, 'batchsize')
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
cmd:option('-maxisz', math.huge, 'max input size allowed during training')
cmd:option('-maxtsz', math.huge, 'max target size allowed during training')
cmd:option('-mintsz', 0, 'min target size allowed during training')
cmd:option('-reload', '', 'reload a particular model')
cmd:option('-reloadarg', false, 'reload argument string')
cmd:option('-continue', '', 'continue a particular model')
cmd:option('-force', false, 'force overwriting experiment')
cmd:option('-noresample', false, 'do not resample training data')
cmd:option('-terrsr', 1, 'train err sample rate (default: each example; 0 is skip)')
cmd:option('-psr', 0, 'perf (statistics) print sample rate (default: only at the end of epochs)')
cmd:option('-replabel', 0, 'replace up to replabel reptitions by additional classes')
cmd:option('-lsm', false, 'add LogSoftMax layer')
cmd:option('-tag', '', 'tag this experiment with a particular name (e.g. "hypothesis1")')

cmd:text()
cmd:text('Data Options:')
cmd:option('-train', '', 'space-separated list of training data')
cmd:option('-valid', '', 'space-separated list of valid data')
cmd:option('-test', '', 'space-separated list of test data')
cmd:option('-samplerate', 16000, 'sample rate (Hz)')
cmd:option('-channels', 1, 'number of input channels')
cmd:option('-dict', 'letters.lst', 'dictionary to use')
cmd:option('-input', 'flac', 'input feature')
cmd:option('-target', 'ltr', 'target feature [phn, ltr, wrd]')

cmd:text()
cmd:text('MFCC Options:')
cmd:option('-mfcc', false, 'use standard htk mfcc features as input')
cmd:option('-pow', false, 'use standard power spectrum as input')
cmd:option('-mfcccoeffs', 13, 'number of mfcc coefficients')
cmd:option('-mfsc', false, 'use standard mfsc features as input')
cmd:option('-melfloor', 0.0, 'specify optional mel floor for mfcc/mfsc/pow')

cmd:text()
cmd:text('Data Augmentation Options:')
cmd:option('-aug', false, 'Enable data augmentations')
cmd:option('-augbendingp', 0, 'Enable pitch bending with given probability')
cmd:option('-augflangerp', 0, 'enable flanger')
cmd:option('-augechorusp', 0, 'enable chorus')
cmd:option('-augechop', 0, 'enable echos')
cmd:option('-augnoisep', 0, 'enable addition of white/brown noise with given probability')
cmd:option('-augcompandp', 0, 'enable compand (may clip!)')
cmd:option('-augspeedp', 0, 'probability with which input speech transformation is applied')
cmd:option('-augspeed', 0, 'variance of input speed transformation')

cmd:text()
cmd:text('Timit-Only Options:')
cmd:option('-seg', false, 'segmentation is given or not')
cmd:option('-dict39', false, 'dictionary with 39 phonemes mode (training -- always for testing)')
cmd:text()

cmd:text()
cmd:text('Misc Options:')
cmd:option('-surround', '', 'surround target with provided label')
cmd:option('-sqnorm', false, 'use square-root when normalizing lr/batchsize/etc...')
cmd:text()

cmd:text('Input shifting Options:')
cmd:option('-shift', 0, 'number of shifts')
cmd:option('-dshift', 0, '# of frames to shift')
cmd:option('-gpushift', false, 'use one GPU per shift')
cmd:text()

local opt = cmd:parse(arg)
local dbg = {} --debugging information (saved first)

local mpi
local mpinn
local mpirank = 1
local mpisize = 1
local function reduce(val)
   return val
end
if opt.mpi then
   mpi = require 'torchmpi'
   mpinn = require 'torchmpi.nn'
   mpi.start(opt.gpu > 0, true)
   mpirank = mpi.rank()+1
   mpisize = mpi.size()
   print(string.format('| MPI #%d/%d', mpirank, mpisize))
   function reduce(val)
      return mpi.allreduce_double(val)/mpisize
   end
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
      continue=true,
      train=true, valid=true, test=true -- DEBUG: FIXME
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
   if not opt.mpi then
      cutorch.setDevice(opt.gpu)
   end
   cutorch.manualSeedAll(opt.seed)
end

local dict = data.newdict{
   path = paths.concat(opt.datadir, opt.dict)
}

if opt.dict39 then
   dict = data.dictcollapsephones{dictionary=dict}
end

if opt.ctc or opt.garbage then
   data.dictadd{dictionary=dict, token="#"} -- blank
end

if opt.replabel > 0 then
   for i=1,opt.replabel do
      data.dictadd{dictionary=dict, token=string.format("%d", i)}
   end
end

opt.nclass = #dict

opt.nchannel = opt.channels

-- if opt.garbage then
--    assert(opt.nstate == 1, 'cannot have garbage and nstate set together')
--    opt.nclass = opt.nclass + 1
-- else
--    opt.nclass = opt.nclass*opt.nstate
-- end

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
local network, kw, dw = netutils.create{
   specs = opt.netspecs,
   gpu = opt.gpu,
   channels = (opt.mfsc and 40 ) or ((opt.pow and 257 ) or (opt.mfcc and opt.mfcccoeffs*3 or opt.channels)), -- DEBUG: UGLY
   nclass = #dict,
   lsm = opt.lsm,
   batchsize = opt.batchsize
}

local zeronet = nn.ZeroNet(kw, dw, opt.nclass)
local netcopy = network:clone() -- pristine stateless copy
opt.kw = kw
opt.dw = dw
local scale
if opt.onorm == 'input' then
   function scale(input, target)
      return opt.sqnorm and math.sqrt(1/input:size(1)) or 1/input:size(1)
   end
elseif opt.onorm == 'target' then
   function scale(input, target)
      return opt.sqnorm and math.sqrt(1/target:size(1)) or 1/target:size(1)
   end
elseif opt.onorm ~= 'none' then
   error('invalid onorm option')
end
print(string.format('| neural network number of parameters: %d', netutils.size(network)))

local function initCriterion(class, ...)
   if opt.batchsize > 0 and class == 'AutoSegCriterion' then
      return nn.BatchAutoSegCriterionC(opt.batchsize, ...)
   elseif opt.batchsize > 0 and opt.mtcrit then
      return nn.MultiThreadedBatchCriterion(opt.batchsize, {'transitions'}, class, ...)
   elseif opt.batchsize > 0 then
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

assert(not(opt.batchsize > 0 and opt.shift > 0), 'Cannot allow both shifting and batching')

if opt.shift > 0 then
   if opt.gpushift then
      network = makeParallel(network, opt.shift)
   else
      network = nn.MapTable(network, {'weight', 'bias'})
      network:resize(opt.shift)
   end
   network = nn.ShiftNet(network, opt.shift, opt.gpushift)
end

local transforms = paths.dofile('transforms.lua')
local remaplabels = transforms.remap{
   uniq = true,
   replabel = opt.replabel > 0 and {n=opt.replabel, dict=dict} or nil
}

local sampler, resample = data.newsampler()
local trainiterator = data.newiterator{
   nthread = opt.nthread,
   closure =
      function()
         local data = paths.dofile('data.lua')
         return data.newdataset{
            names = data.namelist(opt.train),
            opt = opt,
            dict = dict,
            kw = kw,
            dw = dw,
            sampler = sampler,
            mpirank = mpirank,
            mpisize = mpisize,
            aug = opt.aug,
            maxload = opt.maxload
         }
      end
}
local trainsize = trainiterator.execSingle and trainiterator:execSingle('size') or trainiterator:exec('size')

local validiterators = {}
for _, name in ipairs(data.namelist(opt.valid)) do
   validiterators[name] = data.newiterator{
   nthread = opt.nthread,
   closure =
      function()
         local data = paths.dofile('data.lua')
         return data.newdataset{
            names = {name},
            opt = opt,
            dict = dict,
            kw = kw,
            dw = dw,
            mpirank = mpirank,
            mpisize = mpisize,
            maxload = opt.maxloadvalid
         }
      end
   }
end

local testiterators = {}
for _, name in ipairs(data.namelist(opt.test)) do
   testiterators[name] = data.newiterator{
   nthread = opt.nthread,
   closure =
      function()
         local data = paths.dofile('data.lua')
         return data.newdataset{
            names = {name},
            opt = opt,
            dict = dict,
            kw = kw,
            dw = dw,
            mpirank = mpirank,
            mpisize = mpisize,
            maxload = opt.maxloadtest
         }
      end
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
for name, valid in pairs(validiterators) do
   meters.validedit[name] = tnt.EditDistanceMeter()
end

meters.testedit = {}
for name, test in pairs(testiterators) do
   meters.testedit[name] = tnt.EditDistanceMeter()
end

meters.stats = tnt.SpeechStatMeter()


local logfile = torch.DiskFile(
   string.format('%s/log', opt.path),
   (opt.continue == '') and "w" or "rw"
)
local perffile = torch.DiskFile(
   string.format('%s/perf', opt.path),
   (opt.continue == '') and "w" or "rw"
)
do
   log.print2file{file=logfile, date=true, stdout=true}
   local _, header = log.status{meters=meters, state=state, separator=" | ", opt=opt, reduce=reduce, date=true}
   perffile:seekEnd()
   perffile:writeString('# ' .. header .. '\n')
   perffile:synchronize()
end

local function logstatus(meters, state)
   local msgl = log.status{meters=meters, state=state, verbose=true, separator=" | ", opt=opt, reduce=reduce}
   local msgp = log.status{meters=meters, state=state, opt=opt, reduce=reduce, date=true}
   if mpirank == 1 then
      print(msgl)
      logfile:seekEnd()
      logfile:writeString(msgl)
      logfile:writeString("\n")
      logfile:synchronize()

      perffile:seekEnd()
      perffile:writeString(msgp)
      perffile:writeString("\n")
      perffile:synchronize()
   end
end

-- best perf so far on valid datasets
local minerrs = {}
for name, valid in pairs(validiterators) do
   minerrs[name] = math.huge
end

local function save(name, network, best)
   name = name:gsub('/', '#') -- DEBUG: FIXME
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
   if mpirank ~= 1 then
      return
   end

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
   local xlua = require 'xlua'
   local N = iterator.execSingle and iterator:execSingle('size') or iterator:exec('size')
   local n = 0
   return function ()
      if mpirank == 1 then
         n = n + 1
         xlua.progress(n, N)
      end
   end
end

local function map(closure, a)
   if opt.batchsize > 0 then
      local bsz = type(a) == 'table' and #a or a:size(1)
      for k=1,bsz do
         closure(a[k])
      end
   else
      closure(a)
   end
end

local function map2(closure, a, b)
   if opt.batchsize > 0 then
      local bsz = type(a) == 'table' and #a or a:size(1)
      for k=1,bsz do
         closure(a[k], b[k])
      end
   else
      closure(a, b)
   end
end

local function evalOutput(edit, output, target, remaplabels)
   local function evl(o, t)
      edit:add(remaplabels(o), remaplabels(t))
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
      evalOutput(edit, state.network.output, state.sample.target, remaplabels)
   end
   engine:test{
      network = network,
      iterator = iterator
   }
   if progress and mpirank == 1 then
      print()
   end
   return reduce(edit:value())
end

meters.runtime:reset()
local function train(network, criterion, iterator, params, opid)
   local progress
   local engine = tnt.SGDEngine()

   function engine.hooks.onStart(state)
      meters.loss:reset()
      meters.trainedit:reset()
      if opt.mpi then
         mpinn.synchronizeParameters(state.network, true) -- DEBUG: FIXME
         mpinn.synchronizeParameters(state.criterion, true) -- DEBUG: FIXME
      end
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
      meters.networktimer:stop()
      meters.criteriontimer:resume()
      if state.t % opt.terrsr == 0 then
         evalOutput(meters.trainedit, state.network.output, state.sample.target, remaplabels)
      end
      if trainframeerr then
         evalOutput(meters.trainframeerr, state.network.output, state.sample.target, remaplabels)
      end
   end

   function engine.hooks.onBackwardCriterion(state)
      meters.criteriontimer:stop()
      meters.networktimer:resume()
   end

   function engine.hooks.onBackward(state)
      applyClamp()
      meters.networktimer:stop()
      if opt.mpi then
         mpinn.synchronizeGradients(state.network)
         mpinn.synchronizeGradients(state.criterion)
      end
   end

   function engine.hooks.onUpdate(state)
      map(function(out) if out then meters.loss:add(out) end end, state.criterion.output)
      map2(function(i, t) if i then meters.stats:add(i, t) end end, opt.shift > 0 and state.sample.input[1] or state.sample.input, state.sample.target)
      if state.t % opt.psr == 0 and state.t % trainsize ~= 0 then
         if progress then
            print()
         end

         -- print status
         logstatus(meters, state)

         -- save last and best models
         savebestmodels()

         -- Reset average value meters (so that we average over opt.psr steps)
         meters.loss:reset()
         meters.trainedit:reset()
      end
      meters.timer:incUnit()
      meters.sampletimer:incUnit()
      meters.networktimer:incUnit()
      meters.criteriontimer:incUnit()
      meters.sampletimer:resume()
   end

   function engine.hooks.onEndEpoch(state)
      meters.timer:stop()
      meters.sampletimer:stop()
      meters.networktimer:stop()
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
      logstatus(meters, state)

      -- save last and best models
      savebestmodels()

      -- reset meters for next readings
      meters.loss:reset()
      meters.trainedit:reset()
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

local lrnorm = opt.batchsize > 0 and 1/(mpisize*opt.batchsize) or 1/mpisize
lrnorm = opt.sqnorm and math.sqrt(lrnorm) or lrnorm

if not opt.seg and opt.linseg > 0 then
   train(
      opt.linsegznet and zeronet or network,
      lincriterion,
      trainiterator,
      {lr=opt.linlr*lrnorm, lrcriterion=opt.linlrcrit*lrnorm, maxepoch=opt.linseg},
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
   {lr=opt.lr*lrnorm, lrcriterion=opt.lrcrit*lrnorm, maxepoch=opt.iter},
   3
)

if opt.mpi then
   mpi.stop()
end

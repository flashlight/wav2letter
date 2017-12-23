-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local torch = require 'torch'
local g_test = {}
torch.setdefaulttensortype('torch.FloatTensor')
require 'wav2letter'

function g_test.gpumemprof()
   require 'cutorch'
   local d = torch.rand(7000, 7000)
   cutorch.setDevice(1)
   local e11 = d:clone():cuda()
   local e12 = d:clone():cuda()
   local e13 = d:clone():cuda()
   cutorch.setDevice(2)
   local e21 = d:clone():cuda()
   local e22 = d:clone():cuda()
   local e23 = d:clone():cuda()
   cutorch.setDevice(3)
   local e31 = d:clone():cuda()
   local e32 = d:clone():cuda()
   local e33 = d:clone():cuda()
   cutorch.reserveStreams(1)
   cutorch.setStream(1)
   for i = 1, 100 do
      io.write(i .. ' ')
      io.flush()
      e22:copy(e11)
      e32:copy(e22)
      e12:copy(e31)

      e23:copy(e32)
      e33:copy(e11)
      e13:copy(e21)
      cutorch.synchronizeAll()
   end
   io.write('\n')
end

function g_test.convvslinprof()
   require 'cudnn'
   local d = torch.rand(4000, 1, 1000):cuda()
   local e = torch.rand(1000, 4000):cuda()
   local conv = cudnn.SpatialConvolution(4000, 4000, 1, 1, 1, 1):cuda()
   local timer = torch.Timer()
   for i = 1, 100 do
      conv:forward(d)
   end
   print('Conv: ' .. timer:time().real*1000 .. 'ms')
   local lin = nn.Linear(4000, 4000):cuda()
   timer = torch.Timer()
   for i = 1, 100 do
      lin:forward(e)
   end
   print('Linear: ' .. timer:time().real*1000 .. 'ms')
end

function g_test.asglogaddprof()
   local N = 30
   local T = 1000
   local dl = 300
   local noC = false
   local tst2 = nn.AutoSegCriterion(N, false, false, function(input, target) return 1 end, nil, noC)
   tst2.transitions:copy(torch.rand(N, N))
   local d = torch.rand(T, N)
   local dtarget = torch.rand(dl):mul(N):ceil():long()

   --Warmup
   for i = 1, 10 do
      local _loss2 = tst2:forward(d, dtarget)
      local _grad2 = tst2:backward(d, dtarget)
   end
   local timer = torch.Timer()
   for i = 1, 100 do
      local _loss2 = tst2:forward(d, dtarget)
      local _grad2 = tst2:backward(d, dtarget)
   end
   print('Time elapsed: ' .. timer:time().real .. ' seconds')
end

function g_test.batchasglogaddprof()
   local N = 30
   local T = 10000
   local dl = 1000
   local B = 20
   local tst2 = nn.BatchAutoSegCriterionC(B, N, false, false, function(input, target) return 1 end)
   local d = {}
   for i = 1, B do
      d[i] = torch.rand(T, N)
   end
   local dtarget = {}
   for i = 1, B do
      dtarget[i] = torch.rand(dl):mul(N):ceil():long()
   end

   local timer = torch.Timer()
   local _loss2 = tst2:forward(d, dtarget)
   local _grad2 = tst2:backward(d, dtarget)
   print('Time elapsed: ' .. timer:time().real .. ' seconds')
end

function g_test.autosegvsbaidu()
   require 'warpctc'
   -- local A = {28, 5000} --N
   -- local T = {150}
   -- local L = {40, 20} --dl

   local function header(T, N, TT)
      print(string.format("*T*=%d, *N*=%d, *TT*=%d", T, N, TT))
      print("\\begin{tabular}{ Group | S | time(ms)}")
   end

   local function footer()
      print("\\end{tabular}")
      io.write("\n")
      io.flush()
   end

   local samples = 2000
   local function benchFair(bs, T, N, dl)
      local tst2 = nn.BatchAutoSegCriterionC(bs, N, false, false, function(input, target) return 1 end)
      local d
      local dtarget
      if bs > 1 then
         d = {}
         dtarget = {}
         for i = 1, bs do
            d[i] = torch.rand(T, N)
            dtarget[i] = torch.rand(dl):mul(N):ceil():long()
         end
      else
         d = {torch.rand(T, N)}
         dtarget = {torch.rand(dl):mul(N):ceil():long()}
      end
      local function execOne()
         tst2:forward(d, dtarget)
         tst2:updateGradInput(d, dtarget)
      end
      execOne()
      local timer = torch.Timer()
      for i = 1, samples do
         execOne()
      end
      print(string.format('FAIR & %d & %3.2f', bs, timer:time().real*1000/samples))
   end
   local function benchBaidu(bs, T, N, dl, gpu)
      local d
      local dg
      local dtarget
      local sizes = {}
      for i = 1, bs do
         sizes[i] = T
      end
      if bs > 1 then
         d = torch.rand(bs* T, N)
         dg = torch.rand(bs* T, N)
         dtarget = {}
         for i = 1, bs do
            dtarget[i] = {}
            for j = 1, dl do
               dtarget[i][j] = torch.random(1, N)
            end
         end
      else
         d = torch.rand(T, N)
         dg = torch.rand(T, N)
         dtarget = {}
         dtarget[1] = {}
         for j = 1, dl do
            dtarget[1][j] = torch.random(1, N)
         end
      end
      if gpu then
         d = d:cuda()
         dg = dg:cuda()
      end
      local baidu = gpu and gpu_ctc or cpu_ctc
      local function execOne()
         baidu(d, dg, dtarget, sizes)
      end
      execOne()
      local timer = torch.Timer()
      for i = 1, samples do
         execOne()
      end
      if gpu then
         print(string.format('Baidu GPU & %d & %3.2f', bs, timer:time().real*1000/samples))
      else
         print(string.format('Baidu CPU & %d & %3.2f', bs, timer:time().real*1000/samples))
      end
   end


   local function makeTable(T, N, dl)
      header(T, N, dl)
      for _,bs in pairs({1, 4, 8}) do
         bs = tonumber(bs)
         benchFair(bs, T, N, dl)
         benchBaidu(bs, T, N, dl, false)
         benchBaidu(bs, T, N, dl, true)
      end
      footer()
   end

   makeTable(150, 28, 40)
--   makeTable(500, 30, 200)
   makeTable(700, 30, 200)
   --makeTable(1500, 30, 400)
end

local cmd = torch.CmdLine()
cmd:option('-t', '')
local opt = cmd:parse(arg)

if opt.t ~= '' then
   g_test[opt.t]()
else
   g_tester:add(g_test)
   g_tester:run()
end

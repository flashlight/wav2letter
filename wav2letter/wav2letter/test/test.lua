local torch = require 'torch'
local gtn   = require 'gtn'
local nn    = require 'nn'
local utils = require 'wav2letter.utils'
local tnt = require 'torchnet'
require 'wav2letter'
--jac5 fails for 1111, 2222, 3333 with required precision of 1e-2
torch.manualSeed(80)
torch.setdefaulttensortype('torch.FloatTensor')
local g_test = {}
local g_tester = torch.Tester()
local g_log_prec = 1e-3
local g_prec = 1e-5

local function to_logprob(input)
    assert(input:dim() == 2, 'Can only probabilitify 2d tensors')
    local sum = input:sum(2)
    local invs = sum:clone():fill(1):cdiv(sum)
    input:cmul(torch.repeatTensor(invs, 1, input:size(2)))
    return input:log()
end

local function print2d(t, name)
   print(name)
   --print(t)
   for i = 1, t:size(1) do
      for j = 1, t:size(2) do
         io.write(string.format("%4.14f", t[i][j]) .. ' ')
      end
      io.write('\n')
   end
end

local function criterionJacobianTest1DTable(B, input, target, verbose, name, g_cri)
   name = name or 'ConnectionistTemporalCriterion'
   local g_cri = g_cri or nn[name](input:size(2))
   local eps = 1e-3 -- beware, small gives crazy errors with logadd!
   local _ = g_cri:forward(input, target)
   local gw = g_cri:backward( input, target)

   for kk = 1, B do
      local dfdx = gw[kk]
      -- for each input perturbation, do central difference
      local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
      local centraldiff_dfdx_s = centraldiff_dfdx:storage()
      local w_s = input[kk]:storage()
      for i=1,w_s:size() do
         -- f(xi + h)
         w_s[i] = w_s[i] + eps
         local fx1 = g_cri:forward(input, target)[kk]
         -- f(xi - h)
         w_s[i] = w_s[i] - 2*eps
         local fx2 = g_cri:forward(input, target)[kk]
         -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
         local cdfx = (fx1 - fx2) / (2*eps)
         -- store f' in appropriate place
         centraldiff_dfdx_s[i] = cdfx
         -- reset w_s[i]
         w_s[i] = w_s[i] + eps
      end
      -- print2d(centraldiff_dfdx, 'centraldiff_dfdx')

      -- compare centraldiff_dfdx with :backward()
      local err = (centraldiff_dfdx - dfdx):abs():max()
      assert(err < 1e-2, 'Jacobian fails ' .. err)
   end
   if verbose then
      local f = io.open('graph_bw.dot', 'w')
      f:write(gtn.GTN.dot(g_cri.g, false))
      f:close()
      os.execute("dot -Tsvg graph_bw.dot -o graph_bw.svg")
   end
end


local function criterionJacobianTest1D(input, target, verbose, name, g_cri)
   name = name or 'ConnectionistTemporalCriterion'
   local g_cri = g_cri or nn[name](input:size(2))
   local eps = 1e-3 -- beware, small gives crazy errors with logadd!
   local _ = g_cri:forward(input, target)
   local gw = g_cri:backward(input, target)

   local dfdx = gw
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   local w_s = input:storage()
   for i=1,w_s:size() do
      -- f(xi + h)
      w_s[i] = w_s[i] + eps
      local fx1 = g_cri:forward(input, target)
      -- f(xi - h)
      w_s[i] = w_s[i] - 2*eps
      local fx2 = g_cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset w_s[i]
      w_s[i] = w_s[i] + eps
   end
   -- print2d(centraldiff_dfdx, 'centraldiff_dfdx')

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   if verbose then
      local f = io.open('graph_bw.dot', 'w')
      f:write(gtn.GTN.dot(g_cri.g, false))
      f:close()
      os.execute("dot -Tsvg graph_bw.dot -o graph_bw.svg")
   end
   assert(err < 1e-2, 'Jacobian fails ' .. err)
   return err
end

local function simple_test(input, target, fw_p, verbose)
   local g_cri = nn.ConnectionistTemporalCriterion(input:size(2))
   local cri_fw_np = torch.zeros(1):add(g_cri:updateOutput(input, target))
   local log_fw_p = torch.zeros(1):add(math.log(fw_p))
   local msg = string.format('Failed simple test: cri: %f vs. %f', cri_fw_np[1], log_fw_p[1])
   if verbose then
      local f = io.open('graph_fw.dot', 'w')
      f:write(gtn.GTN.dot(g_cri.g, false))
      f:close()
      os.execute("dot -Tsvg graph_fw.dot -o graph_fw.svg")
   end
   --Criterion outputs negative log probability
   g_tester:assertTensorEq(log_fw_p, -cri_fw_np, g_log_prec, msg)
end

--Succeed if forward fails assertion
--TODO: Find cleaner way of doing this
local function fw_error_test(input, target, g_cri)
   local err = pcall(function () return g_cri:updateOutput(input, target) end)
   if err then
      g_tester:assert(1 == 0, "Forward didn't raise error")
   else
      --Maintain correct count of asserts
      g_tester:assert(0 == 0, "Forward did raise error")
   end
end

local function mapping_test(T, N)
   local g_cri = nn.ConnectionistTemporalCriterion()
   local x = torch.Tensor(T,N)
   local s = x:storage()
   for i=1,s:size() do -- fill up the Storage
      s[i] = i
   end
   local input = x
   --input = torch.zeros(3, 2):add(1):div(2):log()
   local target = torch.zeros(1):add(1)
   g_cri:updateOutput(input, target)
   local f = io.open('graph_fw.dot', 'w')
   f:write(gtn.GTN.dot(g_cri.g, false))
   f:close()
   os.execute("dot -Tsvg graph_fw.dot -o graph_fw.svg")
   require("fb.debugger").enter()
end

--121 can only be predicted via emitting 121 directly
--Since there are 3 symbols, the probability should be
--1/3^3
function g_test.test1()
   local input = to_logprob(torch.ones(3, 3))
   local target = torch.zeros(3):add(1)
   target[2] = 2
   simple_test(input, target, 1 / 27)
end

--All symbols have probability 0.5 and the target is
--12. The only path that collapses to this is 12
--so the target probabiltiy is 1/2^2.
function g_test.test2()
   local input = torch.zeros(2, 3):add(1):div(2):log()
   local target = torch.zeros(2):add(1)
   target[2] = 2
   simple_test(input, target, 0.25)
end

--Here we setup the input to only allow a single non-zero prob path
--The -infs are created on purpose and should be ignored
--by the algorithm (they should only be used by inactive paths)
function g_test.test3()
   local input = torch.zeros(3, 3)
   input[1][1] = 1
   input[2][2] = 1
   input[3][1] = 1
   input = to_logprob(input)
   local target = torch.zeros(3):add(1)
   target[2] = 2
   simple_test(input, target, 1)
end

function g_test.test4()
   local input = torch.zeros(1, 3):add(1):div(3):log()
   local target = torch.zeros(1):add(1)
   simple_test(input, target, 1 / 3)
end

--There are 6 paths that produce 1.
function g_test.test5()
   local input = to_logprob(torch.ones(3, 3))
   local target = torch.zeros(1):add(1)
   simple_test(input, target, 6/27)
end

--Empty target tensor
function g_test.test_edge1()
   local input = to_logprob(torch.rand(20, 10))
   local target = torch.Tensor()
   fw_error_test(input, target)
end

--Empty input tensor
function g_test.test_edge2()
   local input = torch.Tensor()
   local target = torch.zeros(3):add(1)
   fw_error_test(input, target)
end

--Target label out of range
function g_test.test_error1()
   local input = torch.zeros(3, 2):add(1):div(2):log()
   local target = torch.zeros(2):add(3)
   fw_error_test(input, target)
end


--Jacobian test on small random input
function g_test.test_jac1()
   local input = to_logprob(torch.rand(3, 3))
   --The highest symbol may only be 9
   local target = torch.zeros(3):add(1)
   target[2] = 2
   criterionJacobianTest1D(input, target, false)
end

--Jacobian test on small random input
function g_test.test_jac1_ones()
   local input = to_logprob(torch.ones(3, 3))
   --The highest symbol may only be 9
   local target = torch.zeros(1):add(1)
   criterionJacobianTest1D(input, target, false)
end

--Jacobian test on small random input
function g_test.test_jac2()
   local input = to_logprob(torch.rand(3, 3))
   --The highest symbol may only be 9
   local target = torch.zeros(1):add(1)
   criterionJacobianTest1D(input, target, false)
end

--Jacobian test on medium random input
function g_test.test_jac3()
   local input = to_logprob(torch.rand(20, 10))
   --The highest symbol may only be 9
   local target = torch.rand(5):mul(9):ceil()
   criterionJacobianTest1D(input, target, false)
end

--Jacobian test on larger random input
function g_test.test_jac4()
   local input = to_logprob(torch.rand(20, 80))
   --The highest symbol may only be 9
   local target = torch.rand(10):mul(79):ceil()
   criterionJacobianTest1D(input, target, false)
end

--Jacobian test on smaller larger random input
function g_test.test_jac5()
   local input = to_logprob(torch.rand(25, 90))
   --The highest symbol may only be 9
   local target = torch.rand(10):mul(8):ceil()
   criterionJacobianTest1D(input, target, false)
end

function g_test.viterbi()
   local T = 20
   local N = 10
   local tst1 = nn.FullConnectCriterion(N,
                                       false,
                                       function(input, target) return 2 end)
   tst1.transitions = torch.rand(N, N)
   local tst2 = nn.Viterbi(N, function(input, target) return 2 end)
   tst2.transitions:copy(tst1.transitions)
   for kk = 1, 100 do
      local d = torch.rand(T, N)
      local path1, score1 = tst1:viterbi(d)
      local path2, score2 = tst2:viterbi(d)
      assert(math.abs(score1 - score2) < 2e-5, 'path score fails')
      assert(path1:add(-1, path2):abs():sum() < 1e-4, 'path fails')
   end

end

function g_test.replabel()
   local target = torch.zeros(30):long():add(1)
   local k = 1
   local nclass = 8
   for i = 1, 4 do
      for j = 1, i do
         target[k] = i
         k = k + 1
      end
   end
   for i = 1, 4 do
      for j = i, 4 do
         target[k] = i
         k = k + 1
      end
   end
   local replabel = 3
   nclass = nclass + replabel
   local t = utils.replabel(target, replabel, nclass)
   local tr = utils.invreplabel(t, replabel, nclass)

   g_tester:assertTensorEq(target:narrow(1, 1, tr:size(1)), tr, 1e-6, 'replabel fails')
   target = torch.zeros(10):long():add(1)
   local t = utils.replabel(target, replabel, nclass)
   local tr = utils.invreplabel(t, replabel, nclass)
   g_tester:assertlt(math.abs(t[2] - 9) , 1e-5, 'replabel does not respect class number boundary')
   g_tester:assertTensorEq(target:narrow(1, 1, tr:size(1)), tr, 1e-6, 'replabel fails')
end

local function comp(t1, t2, name)
   print(name)
   for i = 1, t1:size(1) do
      for j = 1, t1:size(2) do
         if t1[i][j] ~= t2[i][j] then
            print(string.format("(i, j) = (%d, %d); %4.14f ~= %4.14f", i, j, t1[i][j], t2[i][j]))
         end
      end
   end
end

function g_test.fallogadd()
   local N = 20
   local T = 20
   local dl = 10
   local tst1 = nn.ForceAlignCriterion (N, false)
   local tst2 = nn.ForceAlignCriterionC(N, false)
   tst1:zeroGradParameters()
   tst2:zeroGradParameters()
   for kk = 1, 20 do
      --print2d(tst1.gtransitions, 'gtrans1 pre')
      --print2d(tst2.gtransitions, 'gtrans2 pre')
      local d1 = torch.rand(T, N)
      local d2 = d1:clone()
      local dtarget1 = torch.rand(dl):mul(N):ceil():long()
      local dtarget2 = dtarget1:clone()

      local loss1 = tst1:forward(d1, dtarget1)
      local grad1 = tst1:backward(d1, dtarget1):clone()
      local trans1 = tst1.transitions:clone()
      local gtrans1 = tst1.gtransitions:clone()

      local loss2 = tst2:forward(d2, dtarget2)
      local grad2 = tst2:backward(d2, dtarget2):clone()
      local trans2 = tst2.transitions:clone()
      local gtrans2 = tst2.gtransitions:clone()

      if kk % 3 == 0 then
         tst1:updateParameters(0.1)
         tst2:updateParameters(0.1)
      end
      if kk % 4 == 0 then
         tst1:zeroGradParameters()
         tst2:zeroGradParameters()
      end


      local loss_err = math.abs(loss1 - loss2)
      if loss_err > 1e-5 then
         print(string.format('loss fails - kk: %d loss1 %4.14f loss2 %4.14f', kk, loss1, loss2))
         --require("fb.debugger").enter()
         error('loss fails')
      end
      local grad_err = grad1:clone():add(-1, grad2:clone()):abs():mean()
      if grad_err > 1e-5 then
         print(string.format('grad fails - kk: %d grad1:sum() %4.14f grad2:sum() %4.14f', kk, grad1:sum(), grad2:sum()))
         print2d(grad1, 'grad1')
         print2d(grad2, 'grad2')
         --require("fb.debugger").enter()
         error('grad fails')
      end
      local grad_trans_err = gtrans1:clone():add(-1, gtrans2):abs():mean()
      if grad_trans_err > 1e-5 then
         print2d(gtrans1, 'gtrans1')
         print2d(gtrans2, 'gtrans2')
         print(string.format('grad trans fails - kk: %d gtrans1:sum() %4.14f gtrans2:sum() %4.14f', kk, gtrans1:sum(), gtrans2:sum()))
         --comp(gtrans1, gtrans2, 'gtrans')
         for i = 1, dtarget1:size(1) do
            io.write(string.format("%4.14f ", dtarget1[i]))
         end
         io.write("\n")
         --require("fb.debugger").enter()
         error('grad trans fails')
      end
      local trans_err = trans1:clone():add(-1, trans2):abs():mean()
      if trans_err > 1e-5 then
         print(string.format('trans fails - kk: %d trans1:sum() %4.14f trans2:sum() %4.14f', kk, trans1:sum(), trans2:sum()))
         --require("fb.debugger").enter()
         error('trans fails')
      end
      T = T + 1
      dl = dl + 1
   end
   tst1 = nn.ForceAlignCriterion (3, false, false)
   tst2 = nn.ForceAlignCriterionC(3, false, false)
   tst1:zeroGradParameters()
   tst2:zeroGradParameters()
   local input1 = torch.rand(7, 3)
   local input2 = input1:clone()
   local target1 = torch.rand(4):mul(3):ceil():long()
   local target2 = target1:clone()
   print("Running Jacobian Test gtn ")
   for i = 1, 62 do --One more and it fails
      io.write(i .. ' ')
      criterionJacobianTest1D(input1, target1, false, 'ForceAlignCriterion', tst1)
      tst1:updateParameters(0.01)
   end
   io.write('\n')
   print("Running Jacobian Test C ")
   for i = 1, 62 do --One more and it fails
      io.write(i .. ' ')
      criterionJacobianTest1D(input2, target2, false, 'ForceAlignCriterionC', tst2)
      tst2:updateParameters(0.01)
   end
   io.write('\n')
end

function g_test.fcclogadd()
   local N = 20
   local T = 20
   local dl = 10
   local tst1 = nn.FullConnectCriterion (N, false)
   local tst2 = nn.FullConnectCriterionC(N, false)
   for kk = 1, 100 do
      local d1 = torch.rand(T, N)
      local d2 = d1:clone()

      local loss1 = tst1:forward(d1)
      local grad1 = tst1:backward(d1, dtarget):clone()
      local trans1 = tst1.transitions:clone()
      local gtrans1 = tst1.gtransitions:clone()

      local loss2 = tst2:forward(d2)
      local grad2 = tst2:backward(d2, dtarget):clone()
      local trans2 = tst2.transitions:clone()
      local gtrans2 = tst2.gtransitions:clone()

      if kk % 3 == 0 then
         tst1:updateParameters(0.1)
         tst2:updateParameters(0.1)
      end
      if kk % 4 == 0 then
         tst1:zeroGradParameters()
         tst2:zeroGradParameters()
      end

      local function print2d(t, name)
         print(name)
         for i = 1, t:size(1) do
            for j = 1, t:size(2) do
               io.write(string.format("%4.14f", t[i][j]) .. ' ')
            end
            io.write('\n')
         end
      end
      --print2d(gtrans1, 'gtrans1')
      --print2d(gtrans2, 'gtrans2')

      --print(tst2.macc)

      local loss_err = math.abs(loss1 - loss2)
      print(string.format("loss1 %f loss2 %f", loss1, loss2))
      --print("loss1")
      --print(loss1)
      --print("loss2")
      --print(loss2)
      if loss_err ~= 0 then
         print(string.format('loss fails - kk: %d loss1 %4.14f loss2 %4.14f', kk, loss1, loss2))
         require("fb.debugger").enter()
         --error('loss fails')
      end
      local grad_err = grad1:clone():add(-1, grad2:clone()):abs():mean()
      if grad_err ~= 0 then
         print(string.format('grad fails - kk: %d grad1:sum() %4.14f grad2:sum() %4.14f', kk, grad1:sum(), grad2:sum()))
         require("fb.debugger").enter()
         --error('grad fails')
      end
      local grad_trans_err = gtrans1:clone():add(-1, gtrans2):abs():mean()
      if grad_trans_err ~= 0 then
         print(string.format('grad trans fails - kk: %d gtrans1:sum() %4.14f gtrans2:sum() %4.14f', kk, gtrans1:sum(), gtrans2:sum()))
         require("fb.debugger").enter()
         --error('grad trans fails')
      end
      local trans_err = trans1:clone():add(-1, trans2):abs():mean()
      if trans_err ~= 0 then
         print(string.format('trans fails - kk: %d trans1:sum() %4.14f trans2:sum() %4.14f', kk, trans1:sum(), trans2:sum()))
         require("fb.debugger").enter()
         --error('trans fails')
      end
      T = T + 1
      dl = dl + 1
   end
   tst1 = nn.FullConnectCriterion (3, false, false)
   tst2 = nn.FullConnectCriterionC(3, false, false)
   local input1 = torch.rand(7, 3)
   local input2 = input1:clone()
   print("Running Jacobian Test gtn ")
   for i = 1, 100 do
      io.write(i .. ' ')
      criterionJacobianTest1D(input1, nil, false, 'FullConnectCriterion', tst1)
   end
   io.write('\n')
   print("Running Jacobian Test C ")
   for i = 1, 100 do
      io.write(i .. ' ')
      criterionJacobianTest1D(input2, nil, false, 'FullConnectCriterionC', tst2)
   end
   io.write('\n')
end

function g_test.asglogadd()
   local N = 10
   local T = 20
   local dl = 10
   local tst1 = nn.AutoSegCriterion(N, false, false, function(input, target) return 1 end, nil, true)
   local tst2 = nn.AutoSegCriterion(N, false, false, function(input, target) return 1 end, nil, false)
   for kk = 1, 100 do
      local d1 = torch.rand(T, N)
      local d2 = d1:clone()
      local dtarget = torch.rand(dl):mul(N):ceil():long()

      local loss1 = tst1:forward(d1, dtarget)
      local grad1 = tst1:backward(d1, dtarget):clone()
      local trans1 = tst1.transitions:clone()
      local gtrans1 = tst1.gtransitions:clone()

      local loss2 = tst2:forward(d2, dtarget)
      local grad2 = tst2:backward(d2, dtarget):clone()
      local trans2 = tst2.transitions:clone()
      local gtrans2 = tst2.gtransitions:clone()

      if kk % 3 == 0 then
         tst1:updateParameters(0.1)
         tst2:updateParameters(0.1)
      end
      if kk % 4 == 0 then
         tst1:zeroGradParameters()
         tst2:zeroGradParameters()
      end

      local loss_err = math.abs(loss1 - loss2)
      local grad_err = grad1:add(-1, grad2):abs():mean()
      local grad_trans_err = gtrans1:clone():add(-1, gtrans2):abs():mean()
      local trans_err = trans1:clone():add(-1, trans2):abs():mean()
      --if loss_err ~= 0 then
      --   print(string.format('loss fails - kk: %d loss1 4.13%f loss2 4.13%f', kk, loss1, loss2))
      --   require("fb.debugger").enter()
      --   error('loss fails')
      --end
      assert(loss_err < 4e-5, 'loss fails')
      assert(grad_err < 1e-5, 'grad fails')
      assert(grad_trans_err < 1e-5, 'grad trans fails')
      assert(trans_err < 1e-5, 'trans fails')
      T = T + 1
      dl = dl + 1
   end
   tst1 = nn.AutoSegCriterion(3, false, false, function(input, target) return 1 end, nil, true)
   tst1:zeroGradParameters()
   local input1 = torch.rand(7, 3)
   local input2 = input1:clone()
   local target1 = torch.rand(4):mul(3):ceil():long()
   local target2 = target1:clone()
   print("Running Jacobian Test gtn ")
   for i = 1, 300 do --One more and it fails
      io.write(i .. ' ')
      criterionJacobianTest1D(input1, target1, false, nil, tst1)
      tst1:updateParameters(0.01)
   end
   io.write('\n')
   tst2 = nn.AutoSegCriterion(3, false, false, function(input, target) return 1 end, nil, false)
   tst2:zeroGradParameters()
   print("Running Jacobian Test C ")
   for i = 1, 300 do
      io.write(i .. ' ')
      criterionJacobianTest1D(input2, target2, false, nil, tst2)
      tst2:updateParameters(0.01)
   end
   io.write('\n')
end

function g_test.swb()
   local data = tnt.IndexedDataset{
      path = '/mnt/vol/gfsai-east/ai-group/datasets/speech/swb-idx',
      fields = {"input", "target", "uttid"},
   }
   local sndfile = require 'sndfile'
   for i=1,10 do
      local sample = data:get(torch.random(1, data:size()))
      local uttid = sample.uttid:storage():string()
      local f = sndfile.SndFile(sample.input:storage())
      local info = f:info()
      assert(info.samplerate == 8000, 'datapoint has wrong sample rate')
      local ff = f:readFloat(f:info().frames)
      f:close()
      local tmpfname = string.format('/tmp/ex%02d.wav', i)
      local out = sndfile.SndFile(tmpfname, 'w', {samplerate=info.samplerate, channels=1, format="WAV", subformat="ULAW"})
      out:writeFloat(ff)
      out:close()
      print(uttid .. '\t' .. tmpfname .. '\t' .. sample.target:clone():storage():string())
   end
end

function g_test.batchasglogadd()
   local N = 3
   local T = 2
   local dl = 1
   local selfB = 4
   local tst1 = nn.BatchCriterion(selfB, {'transitions'}, 'AutoSegCriterion', N, false, false, function(input, target) return 1 end)
   local tst2 = nn.BatchAutoSegCriterionC(selfB, N, false, false, function(input, target) return 1 end)
   for kk = 1, 75 do
      if kk == 5 then
         T = T / 4
         dl = dl / 4
      else
         T = T + 1
         dl = dl + 1
      end
      local d = {}
      local dtarget = {}
      local B = torch.random(1, selfB)
      for i = 1, B do
         d[i] = torch.rand(T, N)
         dtarget[i] = torch.rand(dl):mul(N):ceil():long()
      end
      local loss2 = tst2:forward(d, dtarget)
      if kk % 4 == 0 then
         tst2:zeroGradParameters()
      end
      local grad2 = tst2:backward(d, dtarget)
      local trans2 = tst2.transitions:clone()

      local loss1 = tst1:forward(d, dtarget)
      if kk % 4 == 0 then
         tst1:zeroGradParameters()
      end
      local grad1 = tst1:backward(d, dtarget)
      local trans1 = tst1.modules[1].transitions:clone()

      if kk % 3 == 0 then
         tst1:updateParameters(0.1)
         tst2:updateParameters(0.1)
      end

      for i = 1, B do
         local loss_err = math.abs(loss1[i] - loss2[i])/loss1[i]
         local grad_err = grad1[i]:clone():add(-1, grad2[i]):abs():mean()
--         require("fb.debugger").enter()
         --g_tester:assertlt(loss_err, 1e-3, 'loss fails')
         --g_tester:assertlt(grad_err, 1e-5, 'grad fails')
         assert(loss_err < 1e-3, 'loss fails')
         assert(grad_err < 1e-4, 'grad fails')
      end
      local trans_err = trans1:clone():add(-1, trans2):abs():mean()
      --g_tester:assertlt(trans_err, 1e-5, 'trans fails')
      assert(trans_err < 1e-4, 'trans fails')
   end
   tst2 = nn.BatchAutoSegCriterionC(2, 3, false, false, function(input, target) return 1 end)
   local target2 = {torch.rand(4):mul(3):ceil():long(),
                    torch.rand(4):mul(3):ceil():long()}
   local input2 = {torch.rand(7, 3), torch.rand(7, 3)}
   print("Running Jacobian Test C ")
   for i = 1, 220 do
      io.write(i .. ' ')
      criterionJacobianTest1DTable(2, input2, target2, false, 'BatchAutoSegCriterionC', tst2)
   end
   io.write('\n')
end

function g_test.libriconfig()
   local sndfile = require 'sndfile'
   local opt = {
      ['datadir'] = '/mnt/vol/gfsai-east/ai-group/datasets/speech',
      ['l8khz']   = true,
      ['maxloadseed']   = 1111,
   }
   local function printExamples(opt, prefix, ...)
      local opt = opt
      local flags = {...}
      local flagsstr = ""
      for k, v in pairs(flags) do
         opt[v] = true
         flagsstr = flagsstr .. v .. ','
      end
      local config = paths.dofile(string.format('../../config/%s.lua', 'librispeech'))
      config = config(opt)
      opt.dataspecs = config.specs
      flagsstr = flagsstr:sub(1, #flagsstr-1)
      local dataset = config.traindataset()
      for i = 1, 3 do
         local ind = torch.random(1, dataset:size())
         local dp = dataset:get(ind)
         local t = dp.input
         local ta = dp.target
         local tmp_wavf = '/tmp/example' .. prefix .. i .. '.wav'
         local tmp_wav = sndfile.SndFile(tmp_wavf, 'w', {samplerate=opt.dataspecs.samplerate, channels=1, format="WAV", subformat="PCM16"})
         tmp_wav:writeFloat(t)
         tmp_wav:close()
         print(flagsstr .. '\t' .. ind .. '\t' .. tmp_wavf .. '\t' .. ta:storage():string())
      end
   end
   printExamples(opt, 'libri', 'lsc100', 'l8khz')
   printExamples(opt, 'fisher', 'lfisher')
   printExamples(opt, 'swb', 'lswb')
end

function g_test.messenger()
   local tnt = require 'torchnet'
   local data = tnt.IndexedDataset{
      path = '/mnt/vol/gfsai-east/ai-group/datasets/speech/messenger-idx',
      fields = {"input", "target", "filename"},
   }
   local sndfile = require 'sndfile'
   for i=1,10 do
      local sample = data:get(torch.random(1, data:size()))
      local filename = sample.filename:storage():string()
      local f = sndfile.SndFile(sample.input:storage())
      local info = f:info()
      assert(info.samplerate == 16000)
      local ff = f:readFloat(f:info().frames)
      f:close()
      local tmpfname = string.format('/tmp/exmessenger%02d.wav', i)
      local out = sndfile.SndFile(tmpfname, 'w', {samplerate=info.samplerate, channels=1, format="WAV", subformat="ULAW"})
      out:writeFloat(ff)
      out:close()

      print(filename .. '\t' .. tmpfname .. '\t' .. sample.target:clone():storage():string())
   end
end

function g_test.fisher()
   local data = tnt.IndexedDataset{
      path = '/mnt/vol/gfsai-east/ai-group/datasets/speech/fisher-idx',
      fields = {"input", "target", "uttID"},
   }
   local sndfile = require 'sndfile'
   for i=1,10 do
      local sample = data:get(torch.random(1, data:size()))
      local uttid = sample.uttID:storage():string()
      local f = sndfile.SndFile(sample.input:storage())
      local info = f:info()
      assert(info.samplerate == 8000)
      local ff = f:readFloat(f:info().frames)
      f:close()

      local tmpfname = string.format('/tmp/ex%02d.wav', i)
      local out = sndfile.SndFile(tmpfname, 'w', {samplerate=info.samplerate, channels=1, format="WAV", subformat="ULAW"})
      out:writeFloat(ff)
      out:close()

      local id, s, e = string.match(uttid, '(.+)-(.+)-(.+)')
      print(id .. '\t' .. s .. '\t' .. e .. '\t' .. tmpfname .. '\t' .. sample.target:clone():storage():string())
   end
end

function g_test.mtcrit()
   local function makeCrits(selfB, N, shared, critname, ...)
      local tst1 = nn.BatchCriterion(selfB, shared, critname, N, ...)
      local tst2 = nn.MultiThreadedBatchCriterion(selfB, shared, critname, N, ...)
      if shared then
         for _, v in pairs(shared) do
            tst1[v]:copy(tst1[v]:clone():random())
            tst1[v]:div(tst1[v]:max())
            tst1[v]:add(-0.5):mul(torch.random(1, 10))
         end
         tst2:share(tst1, unpack(shared))
      end
      return {[1] = tst1, [2] = tst2}
   end

   local function makeDp(selfB, N, maxT)
      local d = {}
      local maxT = maxT or 200
      local dtarget = {}
      local B = torch.random(1, selfB)
      local T = torch.random(2, maxT) --Choice of data length
      local dl = torch.random(1, T) --Choice of target length
      for i = 1, B do
         d[i] = torch.rand(T, N)
         dtarget[i] = torch.rand(dl):mul(N):ceil():long()
      end
      return {input = d, target = dtarget}
   end

   local function compareLoss(loss1, loss2)
      for i = 1, #loss1 do
         local loss_err = math.abs(loss1[i] - loss2[i])/loss1[i]
         assert(loss_err == 0 , 'loss fails 1: ' .. loss1[i] .. ' 2: ' .. loss2[i])
      end
   end

   local function compareGrad(grad1, grad2)
      for i = 1, #grad1 do
         local grad_err = grad1[i]:add(-1, grad2[i]):abs():mean()
         assert(grad_err == 0, 'grad fails')
      end
   end

   local function compareShared(shared, crits)
      for _, v in pairs(shared) do
         local share_err = crits[1][v]:clone():add(-1, crits[2][v]:clone()):abs():mean()
         assert(share_err == 0, v .. ' fails')
      end
   end

   local function testFw(crits, dp)
      local loss1 = crits[1]:forward(dp.input, dp.target)
      local loss2 = crits[2]:forward(dp.input, dp.target)
      compareLoss(loss1, loss2)
   end

   local function testFwBw(shared, crits, dp)
      local loss1 = crits[1]:forward(dp.input, dp.target)
      local loss2 = crits[2]:forward(dp.input, dp.target)
      local grad1 = crits[1]:backward(dp.input, dp.target)
      local grad2 = crits[2]:backward(dp.input, dp.target)
      compareLoss(loss1, loss2)
      compareGrad(grad1, grad2)
      compareShared(shared, crits)
   end

   local function upCrits(crits)
      crits[1]:updateParameters(0.01)
      crits[2]:updateParameters(0.01)
   end

   local function zeroCrits(crits)
      crits[1]:zeroGradParameters()
      crits[2]:zeroGradParameters()
   end

   local function testCrit(shared, critname, N, ...)
      local selfB = 2
      local crits = makeCrits(selfB, N, shared, critname, ...)
      for ep = 1, 5 do
         for i = 1, 5 do
            testFw(crits, makeDp(selfB, N, 10))
            testFwBw(shared, crits, makeDp(selfB, N, 10))
            upCrits(crits)
         end
         zeroCrits(crits)
      end
   end

   local N = 35
   testCrit({'transitions'},  'AutoSegCriterionC', N, false, false, function(input, target) return 1 end)
   testCrit({'transitions'},  'AutoSegCriterion',  N, false, false, function(input, target) return 1 end)
   testCrit({'transitions'},  'CrossEntropyForceAlignCriterion',  N, false, false, function(input, target) return 1 end)
   testCrit({}, 'ForceAlignCriterion',  N, false, false, function(input, target) return 1 end)
   testCrit({'transitions'},  'LinearSegCriterion',  N, false, function(input, target) return 1 end, false)
   testCrit({},  'LinearSegCriterion',  N, false, function(input, target) return 1 end, true)
end

function g_test.linseglogadd()
   local N = 20
   local T = 20
   local dl = 10
   local tst1 = nn.LinearSegCriterion(N, false, nil, nil, false)
   local tst2 = nn.LinearSegCriterion(N, false, nil, nil, true)
   tst1:zeroGradParameters()
   tst2:zeroGradParameters()
   for kk = 1, 20 do
      --print2d(tst1.gtransitions, 'gtrans1 pre')
      --print2d(tst2.gtransitions, 'gtrans2 pre')
      local d1 = torch.rand(T, N)
      local d2 = d1:clone()
      local dtarget1 = torch.rand(dl):mul(N):ceil():long()
      local dtarget2 = dtarget1:clone()

      local loss1 = tst1:forward(d1, dtarget1)
      local grad1 = tst1:backward(d1, dtarget1):clone()
      local trans1 = tst1.transitions:clone()
      local gtrans1 = tst1.fcc.gtransitions:clone()

      local loss2 = tst2:forward(d2, dtarget2)
      local grad2 = tst2:backward(d2, dtarget2):clone()
      local trans2 = tst2.transitions:clone()
      local gtrans2 = tst2.fcc.gtransitions:clone()

      if kk % 3 == 0 then
         tst1:updateParameters(0.1)
         tst2:updateParameters(0.1)
      end
      if kk % 4 == 0 then
         tst1:zeroGradParameters()
         tst2:zeroGradParameters()
      end


      local loss_err = math.abs(loss1 - loss2)
      if loss_err > 1e-5 then
         print(string.format('loss fails - kk: %d loss1 %4.14f loss2 %4.14f', kk, loss1, loss2))
         --require("fb.debugger").enter()
         error('loss fails')
      end
      local grad_err = grad1:clone():add(-1, grad2:clone()):abs():mean()
      if grad_err > 1e-5 then
         print(string.format('grad fails - kk: %d grad1:sum() %4.14f grad2:sum() %4.14f', kk, grad1:sum(), grad2:sum()))
         print2d(grad1, 'grad1')
         print2d(grad2, 'grad2')
         --require("fb.debugger").enter()
         error('grad fails')
      end
      local grad_trans_err = gtrans1:clone():add(-1, gtrans2):abs():mean()
      if grad_trans_err > 1e-5 then
         print2d(gtrans1, 'gtrans1')
         print2d(gtrans2, 'gtrans2')
         print(string.format('grad trans fails - kk: %d gtrans1:sum() %4.14f gtrans2:sum() %4.14f', kk, gtrans1:sum(), gtrans2:sum()))
         --comp(gtrans1, gtrans2, 'gtrans')
         for i = 1, dtarget1:size(1) do
            io.write(string.format("%4.14f ", dtarget1[i]))
         end
         io.write("\n")
         --require("fb.debugger").enter()
         error('grad trans fails')
      end
      local trans_err = trans1:clone():add(-1, trans2):abs():mean()
      if trans_err > 1e-5 then
         print(string.format('trans fails - kk: %d trans1:sum() %4.14f trans2:sum() %4.14f', kk, trans1:sum(), trans2:sum()))
         --require("fb.debugger").enter()
         error('trans fails')
      end
      T = T + 1
      dl = dl + 1
   end
   tst1 = nn.LinearSegCriterion(3, false, nil, nil, false)
   tst2 = nn.LinearSegCriterion(3, false, nil, nil, true)
   tst1:zeroGradParameters()
   tst2:zeroGradParameters()
   local input1 = torch.rand(7, 3)
   local input2 = input1:clone()
   local target1 = torch.rand(4):mul(3):ceil():long()
   local target2 = target1:clone()
   print("Running Jacobian Test gtn ")
   for i = 1, 62 do --One more and it fails
      io.write(i .. ' ')
      criterionJacobianTest1D(input1, target1, false, 'ForceAlignCriterion', tst1)
      tst1:updateParameters(0.01)
   end
   io.write('\n')
   print("Running Jacobian Test C ")
   for i = 1, 62 do --One more and it fails
      io.write(i .. ' ')
      criterionJacobianTest1D(input2, target2, false, 'ForceAlignCriterionC', tst2)
      tst2:updateParameters(0.01)
   end
   io.write('\n')
end

function g_test.editdistance()
   local utils = require 'wav2letter.utils'
   local edit = utils.editdistance

   local function test(o, t, a)
      local ot = torch.Tensor(o):int()
      local tt = torch.Tensor(t):int()
      -- test symmetry
      assert(edit(ot, tt) == a)
      assert(edit(tt, ot) == a)
   end
   test({}, {}, 0)
   test({}, {1}, 1)
   test({2}, {1}, 1)
   test({1}, {1}, 0)
   test({}, {2, 3}, 2)
   test({1, 2, 3}, {2, 3}, 1)
   test({1, 2}, {1, 2, 3}, 1)
   test({1, 2, 4, 3}, {2, 3}, 2)
   test({4, 3}, {2, 3}, 1)
   test({2, 4}, {2, 3}, 1)
   test({1, 4, 3}, {2, 3}, 2)
   test({1, 2, 3}, {4, 5}, 3)
   test({3, 2, 1}, {1, 2, 3}, 2)
   test({3, 2}, {1, 2}, 1)

   --stresstest (malloc)
   local ot = torch.rand(3000):mul(20):floor():int()
   local tt = torch.rand(400):mul(20):floor():int()
   print(edit(ot, tt))
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

-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'
local gtn = require 'gtn'

require 'torch'
torch.manualSeed(1111)
torch.setdefaulttensortype('torch.FloatTensor')

local T = 19
local N = 17
local isviterbi = true

local emissions = torch.rand(T, N)
local transitions = torch.rand(N, N)
local zero = torch.zeros(1)
local gemissions = torch.rand(T, N)
local gtransitions = torch.rand(N, N)
local gzero = torch.zeros(1)

local emissions_p = emissions:data()
local transitions_p = transitions:data()
local zero_p = zero:data()

local gemissions_p = gemissions:data()
local gtransitions_p = gtransitions:data()
local gzero_p = gzero:data()

local g = gtn.GTN()

print('init nodes')
for t=0,T+1 do
   if t == 0 or t==T+1 then
      g:addNode(zero_p, gzero_p)
   else
      for n=0,N-1 do
         g:addNode(emissions_p+(t-1)*N+n, gemissions_p+(t-1)*N+n)
      end
   end
end

print('init edges')
for t=0,T-1 do
   for n=0,N-1 do
      if t==0 then
         g:addEdge(0, n+1, zero_p, gzero_p)
      else
         for nm1=0,N-1 do
            g:addEdge((t-1)*N+nm1+1, t*N+n+1, transitions_p+(n*N+nm1), gtransitions_p+(n*N+nm1))
         end
      end
   end
end

for nm1=0,N-1 do
   g:addEdge((T-1)*N+nm1+1, T*N+1, zero_p, gzero_p)
end

local function criterionJacobianTest1D(cri, w, gw)
   local eps = 1e-3 -- beware, small gives crazy errors with logadd!
   local _ = cri:forward(isviterbi)
   gw:zero()
   cri:backward(1, isviterbi)
   local dfdx = gw
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   local w_s = w:storage()
   for i=1,w_s:size() do
      -- f(xi + h)
      w_s[i] = w_s[i] + eps
      local fx1 = cri:forward(isviterbi)
      -- f(xi - h)
      w_s[i] = w_s[i] - 2*eps
      local fx2 = cri:forward(isviterbi)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset w_s[i]
      w_s[i] = w_s[i] + eps
   end

--   print((centraldiff_dfdx - dfdx):abs())

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   return err
end

local path = torch.LongTensor(g:nNode())
local score, sz = g:forward(path:data(), isviterbi)
print('output:', score)
gemissions:zero()
gtransitions:zero()
g:backward(1, isviterbi)

if N < 20 and T < 20 then
   if sz then
      print(path:narrow(1, 1, sz))
   end
   print('printing')
   local f = io.open('graph.dot', 'w')
   f:write(g:dot(true)) -- true
   f:close()
   print('jac on emmis', criterionJacobianTest1D(g, emissions, gemissions))
   print('jac on trans', criterionJacobianTest1D(g, transitions, gtransitions))
end

print('forwarding')
local MAXITER = tonumber(arg[1]) or 100
local timer = torch.Timer()
for i=1,MAXITER do
   local cost = g:forward(path:data(), isviterbi)
   if i == 1 or i == MAXITER then
      print(string.format('iter=%d cost=%f', i, cost))
   end
   gzero:zero()
   gtransitions:zero()
   gemissions:zero()
   g:backward(1, isviterbi)
   transitions:add(-0.1, gtransitions)
   emissions:add(-0.1, gemissions)
end
print(string.format('[nex/s = %.2f]', MAXITER/timer:time().real))

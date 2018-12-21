--[[
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
--]]

local speech = require 'speech'

local samplerate = 16000
local fs = samplerate
local tw = 25
local ts = 10
local M = 20
local L = 22
local dev = 2
local mel_floor = 1
local R1 = 0
local R2 = samplerate/2
local alpha = 0.97
local N = 13
local T = 8 -- seconds

local preprocess = speech.Mfcc{
   fs=fs, tw=tw, ts=ts, M=M, N=N, L=L, dev=dev, mel_floor=mel_floor,
   alpha=alpha, R1=R1, R2=R2
}
local o
local x = torch.rand(T * samplerate)
for i=1,10 do
  o = preprocess(x) -- warmup
end
local timer = torch.Timer()
for i=1,100 do
  o = preprocess(x)
end
local t = timer:time().real
print(o:sum())
print(string.format("Average time taken  %.5f milliseconds\n",
    t * 1000.0 / 100));

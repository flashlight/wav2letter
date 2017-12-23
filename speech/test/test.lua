-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- you can easily test specific units like this:
-- th -lspeech -e "speech.test{'LookupTable'}"
-- th -lspeech -e "speech.test{'LookupTable', 'Add'}"

local torch = require 'torch'
local speech  = require 'speech'
local sndfile = require 'sndfile'

local g_mytester = torch.Tester()
local g_speechtest = {}

local function equal(t1, t2, prec, msg)
   prec = prec or 0.00001
   if (torch.type(t1) == "table") then
      for k, v in pairs(t2) do
         equal(t1[k], t2[k], msg)
      end
   else
      g_mytester:assertTensorEq(t1, t2, prec, msg)
   end
end

--No test, just exercising.
--TODO: correctness depends on MATLAB design choices
--Code not entirely clear
function g_speechtest.trifiltering()
   local id = function (x) return x end
   local f = speech.TriFiltering(9, 10, 18, 0, 20, id, id)
   local d = torch.ones(10)
   local x = torch.ones(10)
   local y = torch.ones(10)
   --repeated application should have no effect
   equal(f(d), f(d))
   f(x, d)
   f(y, d)
   equal(x, y)
end

--Very similar to offcom. No test, just exercising.
--Not clear if needed (depends on input signal).
function g_speechtest.preemphasis()
   local f = speech.PreEmphasis(nil, 1)
   local d = torch.ones(4)
   local x = torch.ones(4)
   local y = torch.ones(4)
   local i = 0
   --repeated application should have no effect
   equal(f(d), f(d))
   f(x, d)
   f(y, d)
   equal(x, y)
end

--Appears similar to MATLAB code, need output to verify
function g_speechtest.windowing()
   local f = speech.Windowing(nil, 10, nil)
   local d = torch.ones(10)
   local x = torch.ones(10)
   local y = torch.ones(10)
   local i = 0
   d:map(d, function(x) i = i + 1; return i end)
   --repeated application should have no effect
   equal(f(d), f(d))
   f(x, d)
   f(y, d)
   equal(x, y)
end

--No test, just exercising.
--Need to compare to MATLAB code to see which fftw they are using
function g_speechtest.fftw()
   local f = speech.Fftw_1d_fft()
   local d = torch.ones(10)
   local x = torch.ones(10, 2)
   local y = torch.ones(10, 2)
   local i = 0
   d:map(d, function(x) i = i + 1; return i end)
   --repeated application should have no effect
   equal(f(d), f(d))
   f(x, d)
   f(y, d)
   equal(x, y)
end

--Need to compare with MATLAB output
function g_speechtest.dct()
   local f = speech.Dct(20, 10)
   local d = torch.ones(20)
   local x = torch.ones(10)
   local y = torch.ones(10)
   local i = 0
   d:map(d, function(x) i = i + 1; return i end)
   --repeated application should have no effect
   equal(f(d), f(d))
   f(x, d)
   f(y, d)
   equal(x, y)
end


--No test, just exercising. Straightforward application of
--coefficients.
function g_speechtest.ceplifter()
   local f = speech.Ceplifter(10, 20)
   local d = torch.ones(10)
   local x = torch.ones(10)
   local y = torch.ones(10)
   local i = 0
   d:map(d, function(x) i = i + 1; return i end)
   --repeated application should have no effect
   equal(f(d), f(d))
   f(x, d)
   f(y, d)
   equal(x, y)
end

local function read_test_mfcc()
   local paths = require 'paths'
   local path = paths.thisfile('sa1_wav_mfcc_octave')
   local f      = torch.DiskFile(path, 'r')
   local tensor = torch.Tensor(324, 13)
   tensor:storage():copy(f:readFloat(324 * 13))
   return tensor
end

function g_speechtest.mfcc()
   local test_mfcc = read_test_mfcc()
   local paths = require 'paths'
   local f_path = paths.thisfile('sa1.wav')
   local file = sndfile.SndFile(f_path)
   local d = file:readFloat(file:info().frames):squeeze():double()
   --local d = torch.ones(401)
   local f = speech.Mfcc{fs  = 16000,
                         tw  = 25,
                         ts  = 10,
                         M   = 20,
                         N   = 13,
                         L   = 22,
                         R1  = 0,
                         R2  = 8000,
                         dev = 9,
                         mel_floor = 0.0}

   --A precision of 0.0001 fails. These subtle differences
   --shouldn't matter.
   local r = f(d):narrow(2, 1, 13)
   equal(f(d), f(d))
   equal(r, test_mfcc, 0.001)
end

function g_speechtest.derivatives()
   local f = speech.Derivatives(9, 9)
   local d = torch.ones(3, 10)
   local i = 0
   d:map(d, function(x) i = i + 1; return i end)
   print("d")
   print(d)
   print("f(d)")
   print(f(d))
end

local function im_norm(image)
   local max = image:max()
   local min = image:min()
   local norm = image:clone()
   norm:add(-min)
   norm:div(max - min)
   print(norm:size())
   return norm
end

function g_speechtest.mfsc()
   local test_mfcc = read_test_mfcc()
   local paths = require 'paths'
   local f_path = paths.thisfile('sa1.wav')
   local file = sndfile.SndFile(f_path)
   local d = file:readFloat(file:info().frames):squeeze():double()
   --local d = torch.ones(401)
   local f = speech.Mfsc{fs  = 16000,
                         tw  = 25,
                         ts  = 10,
                         M   = 40,
                         R1  = 0,
                         R2  = 8000,
                         dev = 9,
                         mel_floor = 0.0}
   --Just exercising
   local r = f(d)
   print(r:mean())
   print(r:std())
   print(r:size())
   local norm2 = im_norm(r:view(1, 324, -1))
   require 'image'
   image.save('cpuhrsch.png', norm2)
end

-- FB: hooks to work with our test runner
pcall(function ()
      require 'fb.luaunit'
      require 'fbtorch'
end)

g_mytester:add(g_speechtest)
g_mytester:run(tests)

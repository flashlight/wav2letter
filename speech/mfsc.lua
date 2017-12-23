-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local speech = require('libspeech')

--Can be called as mfcc{fs=16000, tw=25, ...} or via mfcc(conf)
--Allows you to use a config file or write code as configuration
local function Mfsc (options)
   -- check mandatory options
   if type(options.fs) ~= "number" then
     error("no frequency given")
   elseif type(options.tw) ~= "number" then
     error("no frame size")
   elseif type(options.ts) ~= "number" then
     error("no frame stride")
   elseif type(options.M) ~= "number" then
     error("no number of filterbank channels (M) given")
   elseif type(options.mel_floor) ~= "number" then
     error("no mel floor given")
   end

   --Based on MATLAB HTK MFCC and ETSI ES 201 208 v 1.1.3(2003-09)
   --I call the MATLAB HTK implementation HTK in here, but I'm not
   --referencing the actual HTK implementation. This is based on the
   --Octave version of that MATLAB implementation (rastamat).
   --frequency (HZ) of speech signal recording
   local fs     = options.fs
   --frame size (no of samples)
   --the last frame is discarded, if less than the frame size
   local fsize  = torch.round(1e-3*options.tw*fs)
   --frame stride (no of samples)
   local stride = torch.round(1e-3*options.ts*fs)
   --preemphasis filtering coefficient
   local alpha  = options.alpha or 0.97
   --analysis window function handle for framing (hann by default)
   local window = options.window
   --lower cutoff frequency (HZ) for the filterbank
   --ETSI considers the lower to be 64hz, htk 300hz
   --(inferring form the matlab example)
   local f_low  = options.R1 or 64
   --upper cutoff frequency (HZ) for the filterbank
   --ETSI considers this to be half of fs, HTK 3700h
   --(inferring form the matlab example)
   local f_high = options.R2 or fs/2
   --number of filterbank channels
   local nof    = options.M
   --option controlling the size of the mel floor
   local mel_floor = options.mel_floor or 1.0

   --fft size
   --we choose it to be a power of two, since the triangular filter
   --assumes, that the maximum frequency is fs/2 and minimum frequency
   --is 0. The grid is then uniformly spaced along [0, fs/2]. Note that
   --this is different from f_low and f_high (parameter R). This
   --specifies the filter width, not the grid layout.
   --number of unique fft components
   local nfft   = 2^torch.ceil(math.log(fsize)/math.log(2))
   local K      = nfft/2 + 1
   --Do operations in place if applicable
   local inplace = 1
   local pow      = torch.Tensor()
   local fft_res  = torch.Tensor()
   local frames   = torch.Tensor()
   --The following functions are defined in the order they are applied
   --PreEmphasis
   local pe  = speech.PreEmphasis(alpha)
   --Applies window, which is hann by default
   local w   = speech.Windowing(window, fsize, inplace)
   local fft = speech.Fftw_1d_fft()
   --The default trifiler is mel, so not specified
   local triflt = speech.TriFiltering(nof, K, fs, f_low, f_high)

   --input_raw is interpreted as a raw speech signal, which is 1d
   return function (output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)
      --Remove singleton dimension if present
      input = input:squeeze()
      input = pe(input)
      --HTK: Values coming out of rasta treat samples as integers,
      --not range -1..1, hence scale up here to match (approx)
      input:mul(32768)
      --IMPORTANT: Just a different view! Frames may overlap.
      input =  input:unfold(input:dim(), fsize, stride)
      frames = frames:type(input:type())
      frames:resizeAs(input)
      frames:copy(input)
      w(frames)
      --Pads the frames with 0 for fft
      pow = pow:type(input:type())
      pow:resize(frames:size(1), nfft)
      pow[{{}, {1, fsize}}]:copy(frames)
      pow[{{}, {fsize+1, nfft}}]:zero()
      --fft buffer tensor
      fft_res = fft_res:type(input:type())
      fft_res:resize(nfft, 2)
      for i = 1, frames:size(1) do
         --Get complex magnitude to produce power spectrum
         local res = fft(fft_res, pow[i])
         pow[i] = res:norm(2, 2):squeeze()
      end
      --Removes all but the unique parts of the frame (up to K)
      --Comments in HTK suggest to apply a mel floor of 1.0 here
      --i.e. making sure no value is lower than 1.0
      --(log is coming up)
      --0.0 is effectively the smallest value, since this is the
      --power spectrum
      pow = pow:narrow(fft_res:dim(), 1, K):cmax(mel_floor)
      ----Apply tri filter
      output = output:type(input:type())
      output:resize(frames:size(1), nof)
      triflt(output, pow)
      --HTK wants to square this now and then apply the natural
      --logarithm (HTK uses that)
      output:log():mul(2)

      return output
   end
end

speech.Mfsc = Mfsc

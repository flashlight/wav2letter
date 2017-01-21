local speech = require('libspeech')

--Applies simple 1d fft, yields complex result
local function Fftw_1d_fft()
   return function (output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)
      output:resize(input:size(1), 2)
      output:zero()
      input.speech.Fftw_forward(input, output)
      return output
   end
end
--TODO: Separate plan function call (return function with steady plan)
--TODO: Option for optimal plan creation
--TODO: Use rank n function

speech.Fftw_1d_fft = Fftw_1d_fft

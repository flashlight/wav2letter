local speech = require('libspeech')

local function PreEmphasis(alpha)
   alpha = alpha or 0.97
   return function(output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)

      output:resize(input:size(1)):copy(input)
      output[{{2, input:size(1)}}]:add(-alpha,
                                   input[{{1, input:size(1)-1}}])
      return output
   end
end

speech.PreEmphasis = PreEmphasis

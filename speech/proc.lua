local speech = require('libspeech')

local function Proc(output_raw, input_raw)
   local input
   local output
   if not input_raw then
      input  = output_raw
      output = torch.Tensor():type(input:type())
   else
      input  = input_raw
      output = output_raw
   end

   return input, output
end

speech.Proc = Proc

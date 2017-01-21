local speech = require('libspeech')

--N is the size of the input
--Window function needs to take two arguments
--n and N (current sample and max samples)
local function Windowing(window, N, inplace)
   local function hanning (n, N)
      return 0.5*(1 -  math.cos(2 * math.pi * (n - 1) / (N - 1)))
   end
   window = window or hanning
   local coefs = torch.range(1, N)
   coefs:apply(function (n) return window(n, N) end)

   return function (output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)
      coefs = coefs:type(input:type())
      if input:dim() == 1 then
         if inplace then
            input:cmul(coefs)
         else
            output:resize(input:size(1))
            output:copy(input)
            return output:cmul(coefs)
         end
      elseif input:dim() == 2 then
         if inplace then
            input:cmul(coefs:view(1, N):expandAs(input))
         else
            output:resizeAs(input)
            output:copy(input)
            return output:cmul(coefs:view(1, N):expandAs(input))
         end
      else
         error("Unsupported dimension.")
      end
   end
end

speech.Windowing = Windowing

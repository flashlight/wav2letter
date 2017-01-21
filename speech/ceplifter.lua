local speech = require('libspeech')

--ceplifter = @( N, L )( 1+0.5*L*sin(pi*[0:N-1]/L) );
--CC = diag( lifter ) * CC; % ~ HTK's MFCCs
--N is the size of the input
local function Ceplifter(N, L, inplace)
   local coefs = torch.range(0, N - 1)
   coefs:mul(math.pi/L)
   coefs:sin()
   coefs:mul(L/2)
   coefs:add(1)
   coefs[1] = 1
   return function(output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)
      coefs = coefs:type(input:type())
      if input:dim() == 1 then
         if inplace then
            input:cmul(coefs)
         else
            output:resizeAs(input)
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

speech.Ceplifter = Ceplifter

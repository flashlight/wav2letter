local speech = require('libspeech')

--Applies Triangular filter to input.
--nof: number of filters (number of rows of filter)
--fn: number of frequency bins (length of input)
--f_low: lower frequency limit
--f_high: higher frequency limit
--fs: sampling frequency (HZ)
--h2w: hertz to warp scale function handle (e.g. mel scale)
--w2h: inverse of h2w
local function TriFiltering(nof, fn, fs, f_low, f_high, h2w, w2h)
   nof = nof or 20
   f_low = f_low or 0
   f_high = f_high or math.floor(fs / 2)
   --Mel warping functions
   h2w = h2w or function(hz)  return 2595 * math.log10(1 + hz / 700) end
   w2h = w2h or function(mel) return 700  * (math.pow(10, mel / 2595) - 1) end

   local f = torch.linspace(0, torch.floor(fs/2), fn)
   local minmel = h2w(f_low)
   local maxmel = h2w(f_high)
   local c = torch.range(0, nof+1)
   c:mul((maxmel - minmel)/ (nof + 1))
   c:add(minmel)
   c:apply(w2h)

   local H = torch.zeros(nof, fn)
   for m = 1, nof do
      local loslope = f:clone()
      local hislope = f:clone()
      loslope:add(-c[m])
      loslope:div(c[m+1] - c[m])
      hislope:mul(-1)
      hislope:add(c[m+2])
      hislope:div(c[m+2] - c[m+1])
      H[m] = torch.cmin(loslope, hislope):cmax(0)
   end

   return function (output_raw, input_raw)
      local input, output = speech.Proc(output_raw, input_raw)
      H = H:type(input:type())
      --Resize fortunately allocates minimally
      if(input:dim() == 1) then
         output:resize(nof):zero()
         return output:addmv(H, input)
      elseif (input:dim() == 2) then
         output:resize(input:size(1), nof):zero()
         return output:t():addmm(H, input:t())
      else
         error("Unsupported dimension")
      end
   end
end

speech.TriFiltering = TriFiltering

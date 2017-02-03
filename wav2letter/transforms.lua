local speechtransform = require 'wav2letter.transform'
local utils = require 'wav2letter.utils'
local transform = require 'torchnet.transform'
local speech = require'speech'
local sndfile = require 'sndfile'

local function as(opt, field, typename)
   assert(opt[field] ~= nil, 'option ' .. field .. ' not set.')
   assert(type(opt[field]) == typename , 'option ' .. field .. ' of wrong type.')
   return opt[field]
end

local function transforms(opt, aug, threadno)
   local transforms = {__shift=0}
   local usertransforms = {} -- DEBUG: remove!
   threadno = threadno or 0

   local samplerate     = as(opt, "samplerate", "number")
   local bending        = as(opt, "bending", "number")
   local saug           = as(opt, "saug", "number")
   local saugp          = as(opt, "saugp", "number")
   local eaug           = as(opt, "eaug", "boolean")
   local vaug           = as(opt, "vaug", "boolean")
   local caug           = as(opt, "caug", "boolean")
   local noise          = as(opt, "noise", "number")
   local outputexample  = as(opt, "outputexample", "boolean")
   local mfcc           = as(opt, "mfcc", "boolean")
   local mfcc_coefs     = as(opt, "mfcc_coefs", "number")
   local mfcc_mel_floor = as(opt, "mfcc_mel_floor", "number")
   local pow            = as(opt, "pow", "boolean")
   local mfsc           = as(opt, "mfsc", "boolean")
   local inormmax       = as(opt, "inormmax", "boolean")
   local inormloc       = as(opt, "inormloc", "boolean")
   local inkw           = as(opt, "inkw", "number")
   local indw           = as(opt, "indw", "number")
   local innt           = as(opt, "innt", "number")
   local kw             = as(opt, "kw", "number")
   local replabel       = as(opt, "replabel", "number")
   local nclass         = as(opt, "nclass", "number")
   local ctc            = as(opt, "ctc", "boolean")
   local nstate         = as(opt, "nstate", "number")

   function transforms.shift(shift)
      if shift then
         transforms.__shift = shift
      end
      return transforms.__shift
   end

   transforms.input = {}
   if aug then
      table.insert(
         transforms.input,
         function (input)
            local tmp_wavf       = os.tmpname() .. threadno .. '.wav'
            local tmp_wavf_res   = os.tmpname() .. threadno .. '.wav'
            local sec_len = (input:size(1)/samplerate)

            local tmp_wav = sndfile.SndFile(tmp_wavf, 'w', {samplerate=samplerate, channels=1, format="WAV", subformat="PCM16"})
            --tmp_wav:writeByte(input:contiguous():storage())
            tmp_wav:writeFloat(input)
            tmp_wav:close()

            --Starting effects
            local exec_str = string.format("sox %s %s", tmp_wavf, tmp_wavf_res)

            --Bending
            if bending >= 0 and torch.uniform() >= bending then
               exec_str = exec_str .. string.format(" bend 0.0,%f,%f", (torch.uniform() - 0.5)*800, sec_len)
            end

            --Speed
            if saug > 0 and saugp > torch.uniform() then
               local coef = (torch.uniform()-0.5)*(saug) + 1
               exec_str = exec_str .. string.format(" speed %f", coef)
            end

            --Chorus
            if eaug and torch.uniform() > 0.5 then
               exec_str = exec_str .. string.format(" chorus 0.7 0.5 55 0.4 0.25 2 -t")
            end

            --Echos
            --Delay 1 and 2, need to be positive
            if eaug and torch.uniform() > 0.5 then
               exec_str = exec_str .. string.format(" echos %f %f %d 0.25 %d 0.3",
                                  torch.uniform()*0.05 + 0.8,
                                  torch.uniform()*0.05 + 0.7,
                                  torch.floor(torch.uniform() * 100) + 1,
                                  torch.floor(torch.uniform() * 100) + 1)
            end

            --Companding
            if vaug and torch.uniform() > 0.5 then
               exec_str = exec_str .. string.format(" compand 0.3,1 6:-70,-60,-%d -15 -50 0.2", torch.floor(torch.uniform()*60))
            end

            --Flanger
            if caug and torch.uniform() > 0.5 then
               exec_str = exec_str .. string.format(" flanger")
            end

            os.execute(exec_str)
            os.remove(tmp_wavf)

            --Noise
            if noise >= 0 and torch.uniform() >= noise then
               local tmp_wavf_noise = os.tmpname() .. '.wav'
               local tmp_wavf_effect= os.tmpname() .. '.wav'
               if torch.uniform() > 0.5 then
                  exec_str = string.format("sox %s %s synth brownnoise vol %f", tmp_wavf_res, tmp_wavf_noise, torch.uniform()*0.03)
               else
                  exec_str = string.format("sox %s %s synth whitenoise vol %f", tmp_wavf_res, tmp_wavf_noise, torch.uniform()*0.03)
               end
               os.execute(exec_str)
               --Mix to 16 bits (sox produces 32 by default, but timit has 16)
               --TODO Double Check Librispeech bitrate
               exec_str = string.format("sox -m -b 16 %s %s %s", tmp_wavf_res, tmp_wavf_noise, tmp_wavf_effect)
               os.execute(exec_str)
               os.remove(tmp_wavf_noise)
               os.remove(tmp_wavf_res)
               tmp_wavf_res = tmp_wavf_effect
            end

            --Example

            if outputexample then
               local dor = string.format('cp %s %s/local/example%d.wav', tmp_wavf_res, paths.home, threadno)
               os.execute(dor)
            end

            --Read and clean up
            local f = sndfile.SndFile(tmp_wavf_res)
            local res = f:readFloat(f:info().frames)
            os.remove(tmp_wavf_res)
            return res
         end
      )
   end

   -- actual input reading
   table.insert(
      transforms.input,
      usertransforms.input
   )

   -- shift if necessary
   table.insert(
      transforms.input,
      function(input)
         local shift = transforms.__shift
         if shift ~= 0 then
            input = input:narrow(1, 1+shift, input:size(1)-shift):clone()
         end
         return input
      end
   )

   if mfcc then
      table.insert(
         transforms.input,
         speech.Mfcc{ fs = samplerate,
                      tw = 25,
                      ts = 10,
                      M  = 20,
                      N  = mfcc_coefs,
                      L  = 22,
                      R1 = 0,
                      R2 = samplerate/2,
                     dev = 9,
                     mel_floor = mfcc_mel_floor}
      )
   end

   if pow then
      table.insert(
         transforms.input,
         speech.Pow{ fs = samplerate,
                      tw = 25,
                      ts = 10,
                     mel_floor = mfcc_mel_floor}
      )
   end

   if mfsc then
      table.insert(
         transforms.input,
         speech.Mfsc{ fs = samplerate,
                      tw = 25,
                      ts = 10,
                      M  = 40,
                     mel_floor = mfcc_mel_floor}
      )
   end

   if inormmax then
      table.insert(
         transforms.input,
         speechtransform.maxnormalize()
      )
   elseif inormloc then
      table.insert(
         transforms.input,
         speechtransform.localnormalize(inkw, indw, innt)
      )
   else
      table.insert(
         transforms.input,
         transform.normalize()
      )
   end
   table.insert(
      transforms.input,
      speechtransform.pad{
         dim=1,
         size=math.floor(kw/2),
         value=0
      }
   )

   -- target
   transforms.target = {usertransforms.target}

   if replabel > 0 then
      table.insert(
         transforms.target,
         function (target)
            return utils.replabel(target, replabel, nclass - ((ctc and 1) or 0))
         end
      )
   end

   table.insert(
      transforms.target,
      utils.uniq
   )

   if nstate > 1 then
      table.insert(
         transforms.target,
         function(target)
            local nstate = nstate
            local nclassdata = nclass
            local etarget = target.new(target:size(1)*nstate)
            for i=1,target:size(1) do
               for j=1,nstate do
                  etarget[(i-1)*nstate+j] = target[i] + (j-1)*nclassdata
               end
            end
            return etarget
         end
      )
   end

   -- output/target remap for evaluation
   transforms.remap = {}

   if nstate > 1 then
      table.insert(
         transforms.remap,
         function(etarget)
            local target = etarget.new(etarget:size(1))
            local nclassdata = nclass
            for i=1,etarget:size(1) do
               target[i] = (etarget[i]-1) % nclassdata + 1
            end
            return target
         end
      )
   end

   table.insert(
      transforms.remap,
      utils.uniq
   )

   if replabel > 0 then
      table.insert(
         transforms.remap,
         function (target)
            return utils.invreplabel(target, replabel, nclass - ((ctc and 1) or 0))
         end
      )
   end

   table.insert(
      transforms.remap,
      usertransforms.remap
   )

   transforms.input = transform.compose(transforms.input)
   transforms.target = transform.compose(transforms.target)
   transforms.remap = transform.compose(transforms.remap)

   return transforms
end

return transforms

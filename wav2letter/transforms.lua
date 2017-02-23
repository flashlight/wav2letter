local argcheck = require 'argcheck'
local speechtransform = require 'wav2letter.transform'
local utils = require 'wav2letter.utils'
local transform = require 'torchnet.transform'
local speech = require'speech'
local sndfile = require 'sndfile'

local transforms = {}

transforms.input = argcheck{
   noordered = true,
   {name='aug', type='table', opt=true}, -- {samplerate=, bendingp=, speedp=, speed=, chorusp=, echop=, compandp=, flangerp=, noisep=, threadid=}
   {name='mfcc', type='table', opt=true}, -- {samplerate=, coeffs=, melfloor=}
   {name='mfsc', type='table', opt=true}, -- {samplerate=, melfloor=}
   {name='pow', type='table', opt=true}, -- {samplerate=, melfloor=}
   {name='normmax', type='boolean', opt=true}, -- {rep=, dict=}
   {name='normloc', type='table', opt=true}, -- {kw=, dw=, noisethreshold=}
   {name='norm', type='boolean', opt=true},
   {name='pad', type='table', opt=true}, -- {size=, value=}
   call =
      function(aug, mfcc, mfsc, pow, normmax, normloc, norm, pad)
         local transforms = {}

         if aug then
            local samplerate, bendingp, speedp, speed, chorusp, echop, compandp, flangerp, noisep, threadid = argcheck{
               noordered = true,
               {name="samplerate", type="number"},
               {name="bendingp", type="number", default=0},
               {name="speedp", type="number", default=0},
               {name="speed", type="number", default=0},
               {name="chorusp", type="number", default=0},
               {name="echop", type="number", default=0},
               {name="compandp", type="number", default=0},
               {name="flangerp", type="number", default=0},
               {name="noisep", type="number", default=0},
               {name="threadid", type="number", default=0},
            }(aug)

            table.insert(
               transforms,
               function(input)
                  local tmp_wavf = os.tmpname() .. threadid .. '.wav'
                  local tmp_wavf_res   = os.tmpname() .. threadid .. '.wav'
                  local sec_len = (input:size(1)/samplerate)
                  local tmp_wav = sndfile.SndFile(tmp_wavf, 'w', {samplerate=samplerate, channels=1, format="WAV", subformat="PCM16"})
                  tmp_wav:writeFloat(input)
                  tmp_wav:close()

                  -- starting effects
                  local exec_str = string.format("sox %s %s", tmp_wavf, tmp_wavf_res)

                  -- bending
                  if torch.uniform() <= bendingp then
                     exec_str = exec_str .. string.format(" bend 0.0,%f,%f", (torch.uniform() - 0.5)*800, sec_len)
                  end

                  -- speed
                  if speed > 0 and torch.uniform() <= speedp then
                     local coef = (torch.uniform()-0.5)*(speed) + 1
                     exec_str = exec_str .. string.format(" speed %f", coef)
                  end

                  -- chorus
                  if torch.uniform() <= chorusp then
                     exec_str = exec_str .. string.format(" chorus 0.7 0.5 55 0.4 0.25 2 -t")
                  end

                  -- echo
                  -- delay 1 and 2, need to be positive
                  if torch.uniform() <= echop then
                     exec_str = exec_str .. string.format(
                        " echos %f %f %d 0.25 %d 0.3",
                        torch.uniform()*0.05 + 0.8,
                        torch.uniform()*0.05 + 0.7,
                        torch.floor(torch.uniform() * 100) + 1,
                        torch.floor(torch.uniform() * 100) + 1)
                  end

                  -- compand
                  if torch.uniform() <= compandp then
                     exec_str = exec_str .. string.format(" compand 0.3,1 6:-70,-60,-%d -15 -50 0.2", torch.floor(torch.uniform()*60))
                  end

                  -- flanger
                  if torch.uniform() <= flangerp then
                     exec_str = exec_str .. string.format(" flanger")
                  end

                  os.execute(exec_str)
                  os.remove(tmp_wavf)

                  -- noise
                  if torch.uniform() <= noise then
                     local tmp_wavf_noise = os.tmpname() .. threadid .. '.wav'
                     local tmp_wavf_effect= os.tmpname() .. threadid .. '.wav'
                     if torch.uniform() > 0.5 then
                        exec_str = string.format("sox %s %s synth brownnoise vol %f", tmp_wavf_res, tmp_wavf_noise, torch.uniform()*0.03)
                     else
                        exec_str = string.format("sox %s %s synth whitenoise vol %f", tmp_wavf_res, tmp_wavf_noise, torch.uniform()*0.03)
                     end
                     os.execute(exec_str)
                     -- mix to 16 bits (sox produces 32 by default, but timit has 16) DEBUG: is this general?
                     exec_str = string.format("sox -m -b 16 %s %s %s", tmp_wavf_res, tmp_wavf_noise, tmp_wavf_effect)
                     os.execute(exec_str)
                     os.remove(tmp_wavf_noise)
                     os.remove(tmp_wavf_res)
                     tmp_wavf_res = tmp_wavf_effect
                  end

                  -- read and clean up
                  local f = sndfile.SndFile(tmp_wavf_res)
                  local res = f:readFloat(f:info().frames)
                  os.remove(tmp_wavf_res)
                  return res
               end
            )
         end

         if mfcc then
            local samplerate, coeffs, melfloor = argcheck{
               noordered = true,
               {name="samplerate", type="number"},
               {name="coeffs", type="number"},
               {name="melfloor", type="number"},
            }(mfcc)
            table.insert(
               transforms,
               speech.Mfcc{
                  fs = samplerate,
                  tw = 25,
                  ts = 10,
                  M  = 20,
                  N  = coeffs,
                  L  = 22,
                  R1 = 0,
                  R2 = samplerate/2,
                  dev = 9,
                  mel_floor = melfloor
               }
            )
         end

         if mfsc then
            local samplerate, melfloor = argcheck{
               noordered = true,
               {name="samplerate", type="number"},
               {name="melfloor", type="number"},
            }(mfsc)
            table.insert(
               transforms,
               speech.Mfsc{
                  fs = samplerate,
                  tw = 25,
                  ts = 10,
                  M  = 40,
                  mel_floor = melfloor
               }
            )
         end

         if pow then
            local samplerate, melfloor = argcheck{
               noordered = true,
               {name="samplerate", type="number"},
               {name="melfloor", type="number"},
            }(pow)

            table.insert(
               transforms,
               speech.Pow{
                  fs = samplerate,
                  tw = 25,
                  ts = 10,
                  mel_floor = melfloor
               }
            )
         end

         if normmax then
            table.insert(
               transforms,
               speechtransform.maxnormalize()
            )
         end

         if normloc then
            local kw, dw, nt = argcheck{
               noordered = true,
               {name="kw", type="number"},
               {name="dw", type="number"},
               {name="noisethreshold", type="number"},
            }(normloc)
            table.insert(
               transforms,
               speechtransform.localnormalize(kw, dw, nt)
            )
         end

         if norm then
            table.insert(
               transforms,
               transform.normalize()
            )
         end

         if pad then
            local size, value = argcheck{
               noordered = true,
               {name="size", type="number"},
               {name="value", type="number", default=0},
            }(pad)
            table.insert(
               transforms,
               speechtransform.pad{
                  dim=1,
                  size=math.floor(size), -- kw/2
                  value=value
               }
            )
         end

         return transform.compose(transforms)
      end
}

transforms.target = argcheck{
   noordered = true,
   {name='surround', type='number', opt=true},
   {name='replabel', type='table', opt=true}, -- {n=, dict=}
   {name='uniq', type='boolean', opt=true},
   call =
      function(surround, replabel, uniq)
         local transforms = {}

         if surround then
            table.insert(
               transforms,
               function(target)
                  local newtarget = target:clone():resize(target:size(1) + 2)
                  newtarget[1] = surround
                  newtarget[target:size(1) + 2] = surround
                  newtarget:narrow(1, 2, target:size(1)):copy(target)
                  return newtarget
               end
            )
         end

         if replabel then
            local n, dict = argcheck{
               noordered = true,
               {name="n", type="number"},
               {name="dict", type="table"},
            }(replabel)
            local replabels = torch.LongTensor(n)
            for i=1,n do
               replabels[i] = assert(dict['R' .. i], 'label does not exist: R' .. i)
            end
            table.insert(
               transforms,
               function(target)
                  return utils.replabel(target, n, replabels)
               end
            )
         end

         if uniq then
            table.insert(
               transforms,
               utils.uniq
            )
         end

         return transform.compose(transforms)

         -- DEPRECATED
         -- if nstate > 1 then
         --    table.insert(
         --       transforms,
         --       function(target)
         --          local nstate = nstate
         --          local nclassdata = nclass
         --          local etarget = target.new(target:size(1)*nstate)
         --          for i=1,target:size(1) do
         --             for j=1,nstate do
         --                etarget[(i-1)*nstate+j] = target[i] + (j-1)*nclassdata
         --             end
         --          end
         --          return etarget
         --       end
         --    )
         -- end
      end
}

transforms.remap = argcheck{
   noordered = true,
   {name='uniq', type='boolean', opt=true},
   {name='replabel', type='table', opt=true}, -- {n=, dict=}
   call =
      function(uniq, replabel)
         local transforms = {}

         -- if nstate > 1 then
         --    table.insert(
         --       transforms,
         --       function(etarget)
         --          local target = etarget.new(etarget:size(1))
         --          local nclassdata = nclass
         --          for i=1,etarget:size(1) do
         --             target[i] = (etarget[i]-1) % nclassdata + 1
         --          end
         --          return target
         --       end
         --    )
         -- end

         if uniq then
            table.insert(
               transforms,
               utils.uniq
            )
         end

         if replabel then
            local n, dict = argcheck{
               noordered = true,
               {name="n", type="number"},
               {name="dict", type="table"},
            }(replabel)
            local replabels = torch.LongTensor(n)
            for i=1,n do
               replabels[i] = assert(dict['R' .. i], 'label does not exist: R' .. i)
            end
            table.insert(
               transforms,
               function (target)
                  return utils.invreplabel(target, n, replabels)
               end
            )
         end

         return transform.compose(transforms)
      end
}

return transforms

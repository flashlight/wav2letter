-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'torch'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1111)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('SpeechRec (c) Ronan Collobert 2015')
cmd:text()
cmd:text('Arguments:')
cmd:argument('-dir', 'directory with output/sentence archives')
cmd:argument('-name', 'name of the pair to decode')
cmd:text()
cmd:text('Options:')
cmd:option('-maxload', -1, 'max number of testing examples')
cmd:option('-show', false, 'show predictions')
cmd:option('-showletters', false, 'show letter predictions')
cmd:option('-letters', "", 'letters.lst')
cmd:option('-words', "", 'words.lst')
cmd:option('-maxword', -1, 'maximum number of words to use')
cmd:option('-lm', "", 'lm.arpa.bin')
cmd:option('-smearing', "none", 'none, max or logadd')
cmd:option('-lmweight', 1, 'lm weight')
cmd:option('-wordscore', 0, 'word insertion weight')
cmd:option('-silweight', 0, 'weight for sil')
cmd:option('-unkscore', -math.huge, 'unknown (word) insertion weight')
cmd:option('-beamsize', 25000, 'max beam size')
cmd:option('-beamscore', 40, 'beam score threshold')
cmd:option('-forceendsil', false, 'force end sil')
cmd:option('-logadd', false, 'use logadd instead of max')
cmd:option('-nthread', 0, 'number of threads to use')
cmd:option('-sclite', false, 'output sclite format')
cmd:text()

local testopt = cmd:parse(arg)
print(string.format("PARAMETERS: -lmweight %f -wordscore %f -unkscore %f -beamsize %f -beamscore %f -silweight %f\n",
                    testopt.lmweight, testopt.wordscore,
                    testopt.unkscore, testopt.beamsize,
                    testopt.beamscore, testopt.silweight))

local function test(opt, slice, nslice)
   local tnt = require 'torchnet'
   require 'wav2letter'

   local decoder = require 'wav2letter.runtime.decoder'
   decoder = decoder(
      opt.letters,
      opt.words,
      opt.lm,
      opt.smearing,
      opt.maxword
   )

   local __unknowns = {}
   local function funk(word)
      if not __unknowns[word] then
         __unknowns[word] = true
         print(string.format('$ warning: unknown word <%s>', word))
      end
   end

   local function tensor2letterstring(t)
      if t:nDimension() == 0 then
         return ""
      end
      local letters = decoder.letters
      local str = {}
      for i=1,t:size(1) do
         local letter = letters[t[i]]
         assert(letter)
         table.insert(str, letter)
      end
      return table.concat(str)
   end

   local fout = tnt.IndexedDatasetReader{
      indexfilename = string.format("%s/output-%s.idx", opt.dir, opt.name),
      datafilename  = string.format("%s/output-%s.bin", opt.dir, opt.name),
      mmap = true,
      mmapidx = true,
   }
   local transitions = torch.DiskFile(string.format("%s/transitions-%s.bin", opt.dir, opt.name)):binary():readObject()

   local wer = tnt.EditDistanceMeter()
   local iwer = tnt.EditDistanceMeter()
   local ler = tnt.EditDistanceMeter()
   local n1 = 1
   local n2 = opt.maxload > 0 and opt.maxload or fout:size()
   local timer = torch.Timer()

   if slice and nslice then
      local nperslice = math.ceil((n2-n1+1)/nslice)
      n1 = (slice-1)*nperslice+1
      if n1 > n2 then
         n1 = 1 -- beware
         n2 = 0
         print(string.format('[slice %d/%d doing nothing]', slice, nslice))
      else
         n2 = math.min(n1+nperslice-1, n2)
         print(string.format('[slice %d/%d going from %d to %d]', slice, nslice, n1, n2))
      end
   end

   local dopt = {
      lmweight = opt.lmweight,
      wordscore = opt.wordscore,
      unkscore = opt.unkscore,
      beamsize = opt.beamsize,
      beamscore = opt.beamscore,
      forceendsil = opt.forceendsil,
      logadd = opt.logadd,
      silweight = opt.silweight
   }

   local sentences = {}
   for i=n1,n2 do
      local prediction = fout:get(i)
      local targets = prediction.words
      local emissions = prediction.output
      local predictions, lpredictions = decoder(dopt, transitions, emissions)
      -- remove <unk>
      predictions = decoder.removeunk(predictions)
      do
         local utils = require 'wav2letter.utils'
         local lpred = utils.uniq(lpredictions)
         --decoder's lpredictions are 0-based
         local spellings = prediction.spellings:apply(function(x) return x - 1 end)
         ler:add(lpred, spellings)

         local targets = decoder.string2tensor(targets, funk)
         iwer:reset()
         iwer:add(predictions, targets)
         wer:add(predictions, targets)
      end
      if opt.show then
         print(
            string.format(
               "%06d |P| %s\n%06d |T| %s {progress=%03d%% iWER=%06.2f%% sliceWER=%06.2f%%}",
               i,
               decoder.tensor2string(predictions),
               i,
               targets:gsub("^%s+", ""):gsub("%s+$", ""),
               n1 == n2 and 100 or (i-n1)/(n2-n1)*100,
               iwer:value(),
               wer:value()
            )
         )
         sentences[i] = {ref=targets:gsub("^%s+", ""):gsub("%s+$", ""), hyp=decoder.tensor2string(predictions)}
      end
      if opt.showletters then
         print(
            string.format(
               "%06d |L| \"%s\"",
               i,
               tensor2letterstring(lpredictions)
            )
         )
         local _, maxEmissions = torch.max(emissions, 2)
         maxEmissions:apply(function(x) return (x - 1) % emissions:size(2) end)
         maxEmissions = maxEmissions:squeeze()
         print(string.format("%06d |M| \"%s\"", i,
                             tensor2letterstring(maxEmissions)))
      end
   end
   print(string.format("[Memory usage: %.2f Mb]", decoder.decoder:mem()/2^20))
   return wer.sum, wer.n, ler.sum, ler.n, n2-n1+1, sentences, timer:time().real
end

local totalacc = 0
local totaln = 0
local totalseq = 0
local totaltime = 0
local totalleracc = 0
local totallern = 0

local timer = torch.Timer()
local sentences = {}

if testopt.nthread <= 0 then
   totalacc, totaln, totalleracc, totallern, totalseq, sentences, totaltime = test(testopt)
else
   local threads = require 'threads'
   local pool = threads.Threads(testopt.nthread)
   for i=1,testopt.nthread do
      pool:addjob(
         function()
            return test(testopt, i, testopt.nthread)
         end,
         function(acc, n, leracc, lern, seq, subsentences, time)
            totalacc = totalacc + acc
            totaln = totaln + n
            totalleracc = totalleracc + leracc
            totallern = totallern + lern
            totalseq = totalseq + seq
            totaltime = totaltime + time
            for i, p in pairs(subsentences) do
               assert(not sentences[i])
               sentences[i] = p
            end
         end
      )
   end
   pool:synchronize()
end

print(string.format("[Decoded %d sequences in %.2f s (actual: %.2f s)]", totalseq, timer:time().real, totaltime))
print(string.format("[WER on %s = %03.2f%%, LER = %03.2f%%]", testopt.name, totalacc/totaln*100, totalleracc/totallern*100))

if testopt.sclite then
   local fhyp = io.open(string.format("%s/sclite-%s.hyp", testopt.dir, testopt.name), "w")
   local fref = io.open(string.format("%s/sclite-%s.ref", testopt.dir, testopt.name), "w")
   for i, p in ipairs(sentences) do
      fhyp:write(string.format("%s (SPEAKER_%05d)\n", p.hyp, i))
      fref:write(string.format("%s (SPEAKER_%05d)\n", p.ref, i))
   end
end

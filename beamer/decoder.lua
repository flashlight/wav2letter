-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'
local env = require 'beamer.env'
local C = env.C

ffi.cdef[[

typedef struct BMRDecoder_  BMRDecoder;

typedef struct BMRDecoderOptions_ {
  long beamsize; /* max beam size */
  float beamscore; /* max delta score kept in the beam */
  float lmweight; /* weight of lm */
  float wordscore; /* score for inserting a word */
  float unkscore; /* score for inserting an unknown */
  int forceendsil; /* force ending in a sil? */
  int logadd; /* use logadd instead of max when merging same word hypothesis */
  float silweight; /* weight of silence */
} BMRDecoderOptions;

BMRDecoder* BMRDecoder_new(BMRTrie *lexicon, BMRLM *lm, long sil, BMRTrieLabel unk);
void BMRDecoder_decode(BMRDecoder *decoder, BMRDecoderOptions *opt, float *transitions, float *emissions, long T, long N, long *nhyp_, float *scores_, long *llabels_, long *labels_);
long BMRDecoder_mem(BMRDecoder *decoder);
void BMRDecoder_free(BMRDecoder *decoder);
void BMRDecoder_settoword(BMRDecoder *decoder, const char* (*toword)(long)); /* for debugging purposes */

]]

local mt = {}

function mt:__new(lexicon, lm, sil, unk)
   assert(unk.lm and unk.usr)
   local self = C.BMRDecoder_new(lexicon, lm, sil, unk)
   if self == nil then
      error('not enough memory')
   end
   ffi.gc(self, C.BMRDecoder_free)
   return self
end

function mt:settoword(toword)
   C.BMRDecoder_settoword(self, toword)
end

local decodeopts = {beamsize=true, beamscore=true, lmweight=true, wordscore=true, unkscore=true, forceendsil=true, logadd=true, silweight=true}
local nhyp_p = ffi.new('long[1]')
function mt:decode(opt, transitions, emissions, scores_, llabels_, labels_)
   local T = emissions:size(1)
   local N = emissions:size(2)
   local B = opt.beamsize

   -- check options
   assert(opt.beamsize, 'opt: beamsize must be specified')
   assert(opt.beamscore, 'opt: beamscore must be specified')
   assert(opt.lmweight, 'opt: lmweight must be specified')
   for k,v in pairs(opt) do
      if not decodeopts[k] then
         error(string.format('opt: unknown option <%s>', k))
      end
   end
   opt = ffi.new('BMRDecoderOptions', opt)

   scores_ = scores_ or torch.FloatTensor()
   llabels_ = llabels_ or torch.LongTensor()
   labels_ = labels_ or torch.LongTensor()
   scores_:resize(B)
   llabels_:resize(B, T+2)
   labels_:resize(B, T+2)

   assert(transitions:size(1) == N)
   assert(transitions:size(2) == N)
   assert(transitions:isContiguous())
   assert(emissions:isContiguous())
   assert(scores_:isContiguous())
   assert(llabels_:isContiguous())
   assert(labels_:isContiguous())

   C.BMRDecoder_decode(self, opt, transitions:data(), emissions:data(), T, N, nhyp_p, scores_:data(), llabels_:data(), labels_:data())

   B = tonumber(nhyp_p[0])
   scores_:resize(B)
   llabels_:resize(B, T+2)
   labels_:resize(B, T+2)

   return scores_, llabels_, labels_
end

function mt:mem()
   return tonumber(C.BMRDecoder_mem(self))
end

env.Decoder = ffi.metatype('BMRDecoder', {__index=mt, __new=mt.__new})

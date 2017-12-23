-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'
local env = require 'beamer.env'
local C = env.C

ffi.cdef[[

typedef struct BMRLM_ BMRLM;
typedef struct BMRLMState_ BMRLMState;

BMRLM& BMRLM_new(const char *path);
long BMRLM_index(BMRLM *lm, const char *word);
BMRLMState& BMRLM_start(BMRLM *lm, int isnull);
BMRLMState& BMRLM_score(BMRLM *lm, BMRLMState *inState, long wordidx, float *score_);
BMRLMState& BMRLM_scorebest(BMRLM *lm, BMRLMState *inState, long* wordidx, long nwords, float *score_, long *nextword_);
void BMRLM_scoreall(BMRLM *lm, BMRLMState *inState, long* words, long nwords, float *score_);
BMRLMState& BMRLM_finish(BMRLM *lm, BMRLMState *inState, float *score_);
float BMRLM_estimate(BMRLM *lm, long *sentence, long size, int isnullstart);
void BMRLM_free(BMRLM *lm);
long BMRLM_mem(BMRLM *lm);
int BMRLMState_compare(BMRLMState *state1_, BMRLMState *state2_);
]]

local mt = {}

function mt:__new(path)
   local self = C.BMRLM_new(path)
   if self == nil then
      error(string.format('could not load LM %s', path))
   end
   ffi.gc(self, C.BMRLM_free)
   return self
end

function mt:compare(state1, state2)
   return C.BMRLMState_compare(state1, state2) == 0
end

function mt:index(word)
   return tonumber(C.BMRLM_index(self, word))
end

function mt:start(isnull)
   return C.BMRLM_start(self, isnull and 1 or 0)
end

local scorep = ffi.new('float[1]')
function mt:score(...)
   local inState, wordidx
   if select('#', ...) == 1 then
      wordidx = select(1, ...)
   elseif select('#', ...) == 2 then
      inState = select(1, ...)
      wordidx = select(2, ...)
   end
   inState = inState or self:start(true)
   if type(wordidx) == 'string' then
      wordidx = self:index(wordidx)
   end
   local outState = C.BMRLM_score(self, inState, wordidx, scorep)
   return outState, scorep[0]
end

local wordidxp = ffi.new('long[1]')
function mt:scorebest(inState, words)
  local nwords = words:size(1)
  assert (words:isContiguous())
  local outState = C.BMRLM_scorebest(self, inState, words:data(), nwords, scorep, wordidxp)
  return outState, scorep[0], tonumber(wordidxp[0])
end

function mt:scoreall(inState, words, scores)
  local nwords = words:size(1)
  assert(words:isContiguous())
  assert(scores:isContiguous())
  assert(scores:size(1) == nwords)
  C.BMRLM_scoreall(self, inState, words:data(), nwords, scores:data())
end


function mt:finish(inState)
   local outState = C.BMRLM_finish(self, inState, scorep)
   return outState, scorep[0]
end

function mt:estimate(t, isnullstart)
   assert(t:isContiguous() and t:nDimension() == 1)
   isnullstart = isnullstart and 1 or 0
   return C.BMRLM_estimate(self, t:data(), t:size(1), isnullstart)
end

function mt:mem()
   return tonumber(C.BMRLM_mem(self))
end

env.LM = ffi.metatype('BMRLM', {__index=mt, __new=mt.__new})

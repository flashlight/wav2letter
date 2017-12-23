-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'
local env = require 'beamer.env'
local C = env.C

ffi.cdef[[

typedef struct BMRTrieNode_  BMRTrieNode;
typedef struct BMRTrie_ BMRTrie;

typedef struct BMRTrieLabel_ {
  int lm; /* lm label */
  int usr; /* usr label */
} BMRTrieLabel;

BMRTrie *BMRTrie_new(long nchildren, long rootidx);
BMRTrieNode* BMRTrie_root(BMRTrie *trie);
BMRTrieNode* BMRTrie_insert(BMRTrie *trie, long *indices, long n, BMRTrieLabel label, float score);
BMRTrieNode* BMRTrie_search(BMRTrie *trie, long *indices, long n);
void BMRTrie_smearing(BMRTrie *trie, int logadd);
void BMRTrie_free(BMRTrie *trie);
long BMRTrie_mem(BMRTrie *trie);

long BMRTrieNode_idx(BMRTrieNode *node);
long BMRTrieNode_nlabel(BMRTrieNode *node);
BMRTrieLabel* BMRTrieNode_label(BMRTrieNode *node, int n);
BMRTrieNode* BMRTrieNode_child(BMRTrieNode *node, long idx);
float BMRTrieNode_score(BMRTrieNode *node, int n);
float BMRTrieNode_maxscore(BMRTrieNode *node);

]]

local mt = {}

function mt:__new(nchildren, rootidx, rootlabel)
   rootidx = rootidx or -1
   local self = C.BMRTrie_new(nchildren, rootidx)
   if rootlabel then
      if C.BMRTrie_insert(self, nil, 0, rootlabel, 0) == nil then
         error('trie: could not insert label (out of memory or BMR_TRIE_MAXLABEL too small)')
      end
   end
   if self == nil then
      error('not enough memory')
   end
   ffi.gc(self, C.BMRTrie_free)
   return self
end

function mt:root()
   local root = C.BMRTrie_root(self)
   return root ~= nil and root or nil
end

function mt:insert(indices, label, score)
   score = score or 0
   assert(indices:nDimension() == 1)
   assert(indices:isContiguous())
   assert(label.lm and label.usr)
   local node = C.BMRTrie_insert(self, indices:data(), indices:size(1), label, score)
   if node == nil then
      error('trie: could not insert label (out of memory or BMR_TRIE_MAXLABEL too small)')
   end
   return node
end

function mt:search(indices)
   assert(indices:nDimension() == 1)
   assert(indices:isContiguous())
   local node = C.BMRTrie_search(self, indices:data(), indices:size(1))
   return node ~= nil and node or nil
end

function mt:smearing(logadd)
   C.BMRTrie_smearing(self, logadd and 1 or 0)
end

function mt:mem()
   return tonumber(C.BMRTrie_mem(self))
end

env.Trie = ffi.metatype('BMRTrie', {__index=mt, __new=mt.__new})

-- BMRTrieNode
local mt = {}

function mt:idx()
   return tonumber(C.BMRTrieNode_idx(self))
end

function mt:label(n)
   if n then
      local label = C.BMRTrieNode_label(self, n)
      return {lm=label.lm, usr=label.usr}
   else
      return tonumber(C.BMRTrieNode_nlabel(self))
   end
end

function mt:score(n)
   if n then
      return C.BMRTrieNode_score(self, n)
   else
      return tonumber(C.BMRTrieNode_nlabel(self))
   end
end

function mt:maxscore()
   return C.BMRTrieNode_maxscore(self)
end

function mt:child(label)
   local node = C.BMRTrieNode_child(self, label)
   return node ~= nil and node or nil
end

ffi.metatype('BMRTrieNode', {__index=mt})

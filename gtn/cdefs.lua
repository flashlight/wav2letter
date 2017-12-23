-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'
local gtn = require 'gtn.env'

ffi.cdef[[

typedef struct GTNGraph_ GTNGraph;

GTNGraph& GTNGraph_new();
void GTNGraph_getNode(GTNGraph *gtn, long idx, float **score, float **gradScore, char *isActive);
long GTNGraph_nNode(GTNGraph *gtn);
long GTNGraph_nEdge(GTNGraph *gtn);
void GTNGraph_free(GTNGraph *gtn);
long GTNGraph_addNode(GTNGraph *gtn, float *score, float* gradScore);
long GTNGraph_addEdge(GTNGraph *gtn, long srcIdx, long dstIdx, float *score, float* gradScore);
char GTNGraph_isActive(GTNGraph *gtn, long idx);

float GTNGraph_forward_logadd(GTNGraph* gtn, long maxidx);
void GTNGraph_backward_logadd(GTNGraph *gtn, float g, long maxidx);

float GTNGraph_forward_max(GTNGraph* gtn, long maxidx, long *path, long *pathsize);
void GTNGraph_backward_max(GTNGraph *gtn, float g, long maxidx);

typedef struct GTNNodeIterator_ GTNNodeIterator;
GTNNodeIterator& GTNNodeIterator_new(GTNGraph *gtn);
void GTNNodeIterator_next(GTNNodeIterator *iterator, long *idx, float **score, float **gradScore, char *isActive);

typedef struct GTNEdgeIterator_ GTNEdgeIterator;
GTNEdgeIterator& GTNEdgeIterator_new(GTNGraph *gtn);
void GTNEdgeIterator_next(GTNEdgeIterator *iterator, long *idx, long *srcidx, long *dstidx, float **score, float **gradScore);

typedef struct GTNNodeEdgeIterator_ GTNNodeEdgeIterator;
GTNNodeEdgeIterator* GTNNodeEdgeIterator_new(GTNGraph *gtn, long nodeidx);
void GTNNodeEdgeIterator_next(GTNNodeEdgeIterator *iterator, long *srcidx, long *dstidx, float **score, float **gradScore);

]]

-- This is (messy) fbcode way. Please do not merge in OSS.
--gtn.C = ffi.load('torch_fb_libgtn')
-- This is the OSS way.
gtn.C = ffi.load(package.searchpath('libgtn', package.cpath))

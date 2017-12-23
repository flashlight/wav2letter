/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef GTN_INC
#define GTN_INC

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "GTNBuffer.h"

#ifdef USE_DOUBLE
#define real double
#define accreal double
#define REAL_MAX DBL_MAX
#else
#define real float
#define accreal double
#define REAL_MAX DBL_MAX
#endif

typedef struct GTNNode_
{
  struct GTNEdge_ *edge;
  struct GTNEdge_ *maxEdge;
  long idx;

  real *score;
  real *gradScore;
  accreal accScore;
  accreal accGradScore;

  char isActive;

} GTNNode;

typedef struct GTNEdge_
{
  struct GTNEdge_ *next;
  struct GTNNode_ *srcNode;
  struct GTNNode_ *dstNode;
  long idx;

  real *score;
  real *gradScore;
  accreal accGradScore;

} GTNEdge;

typedef struct GTNGraph_
{
  GTNBuffer *nodes;
  GTNBuffer *edges;

} GTNGraph;

GTNGraph* GTNGraph_new();
void GTNGraph_getNode(GTNGraph *gtn, long idx, real **score, real **gradScore, char *isActive);
long GTNGraph_nNode(GTNGraph *gtn);
long GTNGraph_nEdge(GTNGraph *gtn);
void GTNGraph_free(GTNGraph *gtn);
long GTNGraph_addNode(GTNGraph *gtn, real *score, real* gradScore);
long GTNGraph_addEdge(GTNGraph *gtn, long srcIdx, long dstIdx, real *score, real* gradScore);
void GTNGraph_markActive(GTNGraph *gtn, long maxidx);
char GTNGraph_isActive(GTNGraph *gtn, long idx);

#endif

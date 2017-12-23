/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "GTNMax.h"

real GTNGraph_forward_max(GTNGraph* gtn, long maxidx, long *path, long *pathsize)
{
  long nnodes = GTNGraph_nNode(gtn);
  nnodes = (maxidx >= 0 && maxidx < nnodes ? maxidx+1 : nnodes);

  GTNGraph_markActive(gtn, maxidx);

  {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, 0);
    node->maxEdge = NULL;
    node->accScore = node->score[0];
  }

  for(long t = 1; t < nnodes; t++) {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, t);
    if(node->isActive) {
      /* finding max edge */
      GTNEdge *edge = node->edge;
      GTNEdge *maxEdge = NULL;
      accreal maxScore = -REAL_MAX;
      while(edge) {
        accreal z = edge->score[0] + edge->srcNode->accScore;
        if(z > maxScore) {
          maxScore = z;
          maxEdge = edge;
        }
        edge = edge->next;
      }
      node->maxEdge = maxEdge;
      node->accScore = maxScore + node->score[0];
    }
  }

  if(path && pathsize) {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, nnodes-1);
    long size = 0;
    while(node) {
      size++;
      node = (node->maxEdge ? node->maxEdge->srcNode : NULL);
    }

    node = GTNBuffer_idx(gtn->nodes, nnodes-1);
    long idx = 0;
    while(node) {
      path[size - ++idx] = node->idx;
      node = (node->maxEdge ? node->maxEdge->srcNode : NULL);
    }

    *pathsize = size;
  }

  return (real) ((GTNNode*)GTNBuffer_idx(gtn->nodes, nnodes-1))->accScore;
}


void GTNGraph_backward_max(GTNGraph *gtn, real g, long maxidx)
{
  long nnodes = GTNGraph_nNode(gtn);
  nnodes = (maxidx >= 0 && maxidx < nnodes ? maxidx+1 : nnodes);

  GTNNode *node = GTNBuffer_idx(gtn->nodes, nnodes-1);
  while(node) {
    node->gradScore[0] += g;
    if(node->maxEdge) {
      node->maxEdge->gradScore[0] += g;
      node = node->maxEdge->srcNode;
    }
    else
      node = NULL;
  }
}

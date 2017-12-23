/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "GTNLogAdd.h"

#define expm(x)  exp(-(x))

real GTNGraph_forward_logadd(GTNGraph* gtn, long maxidx)
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

      /* summing scores */
      accreal sum = 0;
      edge = node->edge;
      while(edge) {
        sum += expm(maxScore - edge->srcNode->accScore - edge->score[0]);
        edge = edge->next;
      }
      sum = maxScore + log(sum) + node->score[0];
      node->accScore = (real)sum;
    }
  }

  return (real) ((GTNNode*)GTNBuffer_idx(gtn->nodes, nnodes-1))->accScore;
}


void GTNGraph_backward_logadd(GTNGraph *gtn, real g, long maxidx)
{
  /* finding maxScore */
  long nnodes = GTNGraph_nNode(gtn);
  nnodes = (maxidx >= 0 && maxidx < nnodes ? maxidx+1 : nnodes);

  {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, nnodes-1);
    node->accGradScore = 1;
  }

  /* zero */
  for(long t = 0; t < nnodes-1; t++) {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, t);
    node->accGradScore = 0;
  }

  for(long t = nnodes-1; t >= 0; t--) {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, t);
    if(node->isActive) {
      node->gradScore[0] += g*node->accGradScore;

      if(node->maxEdge) {
        accreal sum = 0;
        accreal maxScore = node->maxEdge->score[0] + node->maxEdge->srcNode->accScore;
        GTNEdge *edge = node->edge;
        while(edge) {
          edge->accGradScore = expm(maxScore - edge->srcNode->accScore - edge->score[0]);
          sum = sum + edge->accGradScore;
          edge = edge->next;
        }

        edge = node->edge;
        while(edge) {
          edge->accGradScore = edge->accGradScore / sum;
          edge = edge->next;
        }

        edge = node->edge;
        while(edge) {
          edge->srcNode->accGradScore += node->accGradScore*edge->accGradScore;
          edge->gradScore[0] += (real) edge->accGradScore * g*node->accGradScore;
          edge = edge->next;
        }
      }
    }
  }
}

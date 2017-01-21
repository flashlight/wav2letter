/*
 * (c) 2015 Facebook. All rights reserved.
 * Author: Ronan Collobert <locronan@fb.com>
 *
 */

#include "GTN.h"
#include "GTNBuffer.h"

GTNGraph* GTNGraph_new()
{
  GTNGraph *gtn = malloc(sizeof(GTNGraph));
  if(gtn) {
    gtn->nodes = GTNBuffer_new(sizeof(GTNNode));
    gtn->edges = GTNBuffer_new(sizeof(GTNEdge));
  }
  return gtn;
}

long GTNGraph_nNode(GTNGraph *gtn)
{
  return GTNBuffer_size(gtn->nodes);
}

long GTNGraph_nEdge(GTNGraph *gtn)
{
  return GTNBuffer_size(gtn->edges);
}

void GTNGraph_free(GTNGraph *gtn)
{
  GTNBuffer_free(gtn->nodes);
  GTNBuffer_free(gtn->edges);
  free(gtn);
}

long GTNGraph_addNode(GTNGraph *gtn, real *score, real* gradScore)
{
  long nodeidx = GTNBuffer_size(gtn->nodes);
  if(!GTNBuffer_grow(gtn->nodes, nodeidx+1))
    return -1;
  if(!score || !gradScore)
    return -2;

  GTNNode *node = GTNBuffer_idx(gtn->nodes, nodeidx);
  node->edge = NULL;
  node->maxEdge = NULL;
  node->score = score;
  node->gradScore = gradScore;
  node->accScore = 0;
  node->accGradScore = 0;
  node->idx = nodeidx;

  return nodeidx;
}

void GTNGraph_getNode(GTNGraph *gtn, long idx, real **score, real **gradScore, char *isActive)
{
  GTNNode *node = GTNBuffer_idx(gtn->nodes, idx);
  *score = node->score;
  *gradScore = node->gradScore;
  *isActive = node->isActive;
}

long GTNGraph_addEdge(GTNGraph *gtn, long srcIdx, long dstIdx, real *score, real* gradScore)
{
  long edgeidx = GTNBuffer_size(gtn->edges);

  if(!score || !gradScore)
    return -2;

  if(srcIdx >= dstIdx || srcIdx < 0 || dstIdx >= GTNBuffer_size(gtn->nodes))
    return -3;

  if(!GTNBuffer_grow(gtn->edges, edgeidx+1))
    return -1;

  GTNEdge *edge = GTNBuffer_idx(gtn->edges, edgeidx);
  GTNNode *dstNode = GTNBuffer_idx(gtn->nodes, dstIdx);
  edge->srcNode = GTNBuffer_idx(gtn->nodes, srcIdx);
  edge->score = score;
  edge->gradScore = gradScore;
  edge->accGradScore = 0;
  edge->next = dstNode->edge;
  edge->idx = edgeidx;
  edge->dstNode = dstNode;
  dstNode->edge = edge;

  return edgeidx;
}

void GTNGraph_markActive(GTNGraph *gtn, long maxidx)
{
  long nnodes = GTNGraph_nNode(gtn);
  nnodes = (maxidx >= 0 && maxidx < nnodes ? maxidx+1 : nnodes);

  for(long t = 0; t < nnodes; t++) {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, t);
    node->isActive = 0;
  }

  GTNNode *node = GTNBuffer_idx(gtn->nodes, nnodes-1);
  node->isActive = 1;

  for(long t = nnodes-1; t > 0; t--) {
    GTNNode *node = GTNBuffer_idx(gtn->nodes, t);
    if(node->isActive) {
      GTNEdge *edge = node->edge;
      while(edge) {
        edge->srcNode->isActive = 1;
        edge = edge->next;
      }
    }
  }
}

char GTNGraph_isActive(GTNGraph *gtn, long idx)
{
  GTNNode *node = GTNBuffer_idx(gtn->nodes, idx);
  return node->isActive;
}

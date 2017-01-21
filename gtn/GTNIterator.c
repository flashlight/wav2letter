/*
 * (c) 2015 Facebook. All rights reserved.
 * Author: Ronan Collobert <locronan@fb.com>
 *
 */

#include "GTNIterator.h"

struct GTNNodeIterator_
{
  long idx;
  GTNBuffer *nodes;
};

GTNNodeIterator* GTNNodeIterator_new(GTNGraph *gtn)
{
  GTNNodeIterator *iterator = malloc(sizeof(GTNNodeIterator));
  if(iterator) {
    iterator->nodes = gtn->nodes;
    iterator->idx = -1;
  }
  return iterator;
}

void GTNNodeIterator_next(GTNNodeIterator *iterator, long *idx, real **score, real **gradScore, char *isActive)
{
  if(++iterator->idx < GTNBuffer_size(iterator->nodes)) {
    GTNNode *node = GTNBuffer_idx(iterator->nodes, iterator->idx);
    *idx = iterator->idx;
    *score = node->score;
    *gradScore = node->gradScore;
    *isActive = node->isActive;
  }
  else {
    *idx = -1;
    *score = NULL;
    *gradScore = NULL;
    *isActive = 0;
  }
}

struct GTNEdgeIterator_
{
  long idx;
  GTNBuffer *edges;
};

GTNEdgeIterator* GTNEdgeIterator_new(GTNGraph *gtn)
{
  GTNEdgeIterator *iterator = malloc(sizeof(GTNEdgeIterator));
  if(iterator) {
    iterator->edges = gtn->edges;
    iterator->idx = -1;
  }
  return iterator;
}

void GTNEdgeIterator_next(GTNEdgeIterator *iterator, long *idx, long *srcidx, long *dstidx, real **score, real **gradScore)
{
  if(++iterator->idx < GTNBuffer_size(iterator->edges)) {
    GTNEdge *edge = GTNBuffer_idx(iterator->edges, iterator->idx);
    *idx = iterator->idx;
    *srcidx = edge->srcNode->idx;
    *dstidx = edge->dstNode->idx;
    *score = edge->score;
    *gradScore = edge->gradScore;
  }
  else {
    *idx = -1;
    *srcidx = -1;
    *dstidx = -1;
    *score = NULL;
    *gradScore = NULL;
  }
}

struct GTNNodeEdgeIterator_
{
  GTNEdge *edge;
};

GTNNodeEdgeIterator* GTNNodeEdgeIterator_new(GTNGraph *gtn, long nodeidx)
{
  GTNNodeEdgeIterator *iterator = malloc(sizeof(GTNNodeEdgeIterator));
  GTNNode *node = GTNBuffer_idx(gtn->nodes, nodeidx);
  if(node && iterator) {
    iterator->edge = node->edge;
  }
  return iterator;
}

void GTNNodeEdgeIterator_next(GTNNodeEdgeIterator *iterator, long *srcidx, long *dstidx, real **score, real **gradScore)
{
  if(iterator->edge) {
    GTNEdge *edge = iterator->edge;
    *srcidx = edge->srcNode->idx;
    *dstidx = edge->dstNode->idx;
    *score = edge->score;
    *gradScore = edge->gradScore;
    iterator->edge = iterator->edge->next;
  }
  else {
    *srcidx = -1;
    *dstidx = -1;
    *score = NULL;
    *gradScore = NULL;
  }
}

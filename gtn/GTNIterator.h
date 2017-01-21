/*
 * (c) 2015 Facebook. All rights reserved.
 * Author: Ronan Collobert <locronan@fb.com>
 *
 */

#ifndef GTN_ITERATOR_INC
#define GTN_ITERATOR_INC

#include "GTN.h"

typedef struct GTNNodeIterator_ GTNNodeIterator;
GTNNodeIterator* GTNNodeIterator_new(GTNGraph *gtn);
void GTNNodeIterator_next(GTNNodeIterator *iterator, long *idx, real **score, real **gradScore, char *isActive);

typedef struct GTNEdgeIterator_ GTNEdgeIterator;
GTNEdgeIterator* GTNEdgeIterator_new(GTNGraph *gtn);
void GTNEdgeIterator_next(GTNEdgeIterator *iterator, long *idx, long *srcidx, long *dstidx, real **score, real **gradScore);

typedef struct GTNNodeEdgeIterator_ GTNNodeEdgeIterator;
GTNNodeEdgeIterator* GTNNodeEdgeIterator_new(GTNGraph *gtn, long nodeidx);
void GTNNodeEdgeIterator_next(GTNNodeEdgeIterator *iterator, long *srcidx, long *dstidx, real **score, real **gradScore);

#endif

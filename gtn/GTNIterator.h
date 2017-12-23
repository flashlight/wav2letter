/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
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

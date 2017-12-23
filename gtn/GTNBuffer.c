/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <stdio.h>
#include <stdlib.h>

#include "GTNBuffer.h"

/* yes, we could have make it logarithmic */
/* but, pfff... */
#define GTN_BUFFER_BUCKET_SIZE 1024

struct GTNBuffer_
{
  void **buckets;
  long nbuckets;
  long n;
  long size;
};

GTNBuffer* GTNBuffer_new(long size)
{
  GTNBuffer *buffer = malloc(sizeof(GTNBuffer));
  if(buffer) {
    buffer->buckets = NULL;
    buffer->nbuckets = 0;
    buffer->n = 0;
    buffer->size = size;
  }
  return buffer;
}

void* GTNBuffer_grow(GTNBuffer *buffer, long n)
{
  if(n > buffer->n) {
    while(n > buffer->nbuckets * GTN_BUFFER_BUCKET_SIZE) {
      void *data = realloc(buffer->buckets, (buffer->nbuckets+1)*sizeof(void*));
      if(!data)
        return NULL;
      buffer->buckets = data;
      buffer->buckets[buffer->nbuckets] = malloc(buffer->size*GTN_BUFFER_BUCKET_SIZE);
      if(!buffer->buckets[buffer->nbuckets])
        return NULL;
      buffer->nbuckets++;
    }
    buffer->n = n;
  }
  return GTNBuffer_idx(buffer, n-1);
}

void* GTNBuffer_idx(GTNBuffer *buffer, long idx)
{
  if(idx < 0 || idx >= buffer->n)
    return NULL;

  return buffer->buckets[idx/GTN_BUFFER_BUCKET_SIZE]+buffer->size*(idx % GTN_BUFFER_BUCKET_SIZE);
}

long GTNBuffer_size(GTNBuffer *buffer)
{
  return buffer->n;
}

void GTNBuffer_free(GTNBuffer *buffer)
{
  for(long i = 0; i < buffer->nbuckets; i++)
    free(buffer->buckets[i]);
  free(buffer->buckets);
  free(buffer);
}

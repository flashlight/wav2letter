/* (c) Ronan Collobert 2016, Facebook */

#include <stdlib.h>

#include "BMRBuffer.h"

/* yes, we could have make it logarithmic */
/* but, pfff... */
#define BMR_BUFFER_BUCKET_SIZE 32768

struct BMRBuffer_
{
  void **buckets;
  long nbuckets;
  long n;
  long size;

};

BMRBuffer* BMRBuffer_new(long size)
{
  BMRBuffer *buffer = malloc(sizeof(BMRBuffer));
  if(buffer) {
    buffer->buckets = NULL;
    buffer->nbuckets = 0;
    buffer->n = 0;
    buffer->size = size;
  }
  return buffer;
}

void* BMRBuffer_grow(BMRBuffer *buffer, long n)
{
  if(n > buffer->n) {
    while(n > buffer->nbuckets * BMR_BUFFER_BUCKET_SIZE) {
      void *data = realloc(buffer->buckets, (buffer->nbuckets+1)*sizeof(void*));
      if(!data)
        return NULL;
      buffer->buckets = data;
      buffer->buckets[buffer->nbuckets] = malloc(buffer->size*BMR_BUFFER_BUCKET_SIZE);
      if(!buffer->buckets[buffer->nbuckets])
        return NULL;
      buffer->nbuckets++;
    }
    buffer->n = n;
  }
  return BMRBuffer_idx(buffer, n-1);
}

void* BMRBuffer_idx(BMRBuffer *buffer, long idx)
{
  if(idx < 0 || idx >= buffer->n)
    return NULL;

  return buffer->buckets[idx/BMR_BUFFER_BUCKET_SIZE]+buffer->size*(idx % BMR_BUFFER_BUCKET_SIZE);
}

long BMRBuffer_size(BMRBuffer *buffer)
{
  return buffer->n;
}

void BMRBuffer_reset(BMRBuffer *buffer)
{
  buffer->n = 0;
}

long BMRBuffer_mem(BMRBuffer *buffer)
{
  return buffer->n*buffer->size+buffer->nbuckets*sizeof(void*)+sizeof(BMRBuffer);
}

void BMRBuffer_free(BMRBuffer *buffer)
{
  if(buffer) {
    long i;
    for(i = 0; i < buffer->nbuckets; i++)
      free(buffer->buckets[i]);
    free(buffer->buckets);
    free(buffer);
  }
}

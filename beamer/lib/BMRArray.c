/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "BMRArray.h"
#include <stdlib.h>


#define BMR_ARRAY_INC_SIZE 32768

struct BMRArray_
{
  void **data;
  long size;
  long maxsize;
};

/* Note: beware that qsort works on ** */
#define IDX(III) (arr+(III))
#define SWAP(AAA, BBB) rswap = *IDX(AAA); *IDX(AAA) = *IDX(BBB); *IDX(BBB) = rswap

static void quickselect(void **arr, long size, long k, int (*compar)(const void *elem1_, const void *elem2_))
{
  long P, L, R, i, j;
  void **piv;
  void *rswap;

  L = 0;
  R = size-1;

  do {
    if (R <= L) /* One element only */
      return;

    if (R == L+1) {  /* Two elements only */
      if (compar(IDX(L), IDX(R)) > 0) {
        SWAP(L, R);
      }
      return;
    }

    /* Use median of three for pivot choice */
    P=(L+R)>>1;
    SWAP(P, L+1);
    if (compar(IDX(L+1), IDX(R)) > 0) { SWAP(L+1, R); }
    if (compar(IDX(L), IDX(R)) > 0) { SWAP(L, R); }
    if (compar(IDX(L+1), IDX(L)) > 0) { SWAP(L+1, L); }

    i = L+1;
    j = R;
    piv = IDX(L);
    do {
      do i++; while(compar(IDX(i), piv) < 0);
      do j--; while(compar(IDX(j), piv) > 0);
      if (j < i)
        break;
      SWAP(i, j);
    } while(1);
    SWAP(L, j);

    /* Re-set active partition */
    if (j <= k) L=i;
    if (j >= k) R=j-1;
  } while(1);
}

BMRArray* BMRArray_new(long size)
{
  BMRArray *self = malloc(sizeof(BMRArray));
  if(self) {
    self->size = 0;
    self->maxsize = 0;
    self->data = NULL;
    BMRArray_resize(self, size);
  }
  return self;
}

int BMRArray_resize(BMRArray *self, long size)
{
  long i;
  if(size > self->maxsize) {
    long newmaxsize = (size > self->maxsize + BMR_ARRAY_INC_SIZE ? size : self->maxsize + BMR_ARRAY_INC_SIZE); /* DEBUG: could be exponential */
    self->data = realloc(self->data, newmaxsize*sizeof(void*));
    if(!self->data) {
      self->size = 0;
      self->maxsize = 0;
      return 1;
    }
    self->maxsize = newmaxsize;
  }
  for(i = self->size; i < size; i++)
    self->data[i] = NULL;
  self->size = size;
  return 0;
}

void* BMRArray_get(BMRArray *self, long idx)
{
  if(idx >= self->size)
    return NULL;
  return self->data[idx];
}

void* BMRArray_set(BMRArray *self, long idx, void *elem)
{
  if(idx >= self->size) {
    if(BMRArray_resize(self, idx+1))
      return NULL;
  }
  self->data[idx] = elem;
  return elem;
}

void* BMRArray_add(BMRArray *self, void *elem)
{
  return BMRArray_set(self, BMRArray_size(self), elem);
}

long BMRArray_size(BMRArray *self)
{
  return self->size;
}

void BMRArray_reset(BMRArray *self)
{
  self->size = 0;
}

void BMRArray_free(BMRArray *self)
{
  free(self->data);
  free(self);
}

long BMRArray_mem(BMRArray *self)
{
  return sizeof(void*)*self->maxsize;
}

void BMRArray_sort(BMRArray *self, int (*compar)(const void *elem1_, const void *elem2_))
{
  qsort(self->data, self->size, sizeof(void*), compar);
}

void BMRArray_topksort(BMRArray *self, int (*compar)(const void *elem1_, const void *elem2_), long k)
{
  quickselect(self->data, self->size, k, compar);
}

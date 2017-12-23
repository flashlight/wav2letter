/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef BMR_ARRAY_INC
#define BMR_ARRAY_INC

typedef struct BMRArray_ BMRArray;

BMRArray* BMRArray_new(long size);
int BMRArray_resize(BMRArray *self, long size);
void* BMRArray_get(BMRArray *self, long idx);
void* BMRArray_set(BMRArray *self, long idx, void *elem);
void* BMRArray_add(BMRArray *self, void *elem);
long BMRArray_size(BMRArray *self);
void BMRArray_reset(BMRArray *self);
void BMRArray_free(BMRArray *self);
long BMRArray_mem(BMRArray *self);
void BMRArray_sort(BMRArray *self, int (*compare)(const void *elem1_, const void *elem2_));
void BMRArray_topksort(BMRArray *self, int (*compar)(const void *elem1_, const void *elem2_), long k);

#endif

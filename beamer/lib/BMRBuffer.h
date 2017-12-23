/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef BMR_BUFFER_INC
#define BMR_BUFFER_INC

typedef struct BMRBuffer_ BMRBuffer;

BMRBuffer* BMRBuffer_new(long size);
void* BMRBuffer_grow(BMRBuffer *buffer, long n);
void* BMRBuffer_idx(BMRBuffer *buffer, long idx);
long BMRBuffer_size(BMRBuffer *buffer);
void BMRBuffer_reset(BMRBuffer *buffer);
void BMRBuffer_free(BMRBuffer *buffer);
long BMRBuffer_mem(BMRBuffer *buffer);

#endif

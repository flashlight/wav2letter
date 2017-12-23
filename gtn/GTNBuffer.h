/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef GTN_BUFFER_INC
#define GTN_BUFFER_INC

typedef struct GTNBuffer_ GTNBuffer;

GTNBuffer* GTNBuffer_new(long size);
void* GTNBuffer_grow(GTNBuffer *buffer, long n);
void* GTNBuffer_idx(GTNBuffer *buffer, long idx);
long GTNBuffer_size(GTNBuffer *buffer);
void GTNBuffer_free(GTNBuffer *buffer);

#endif

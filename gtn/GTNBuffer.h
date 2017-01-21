/*
 * (c) 2015 Facebook. All rights reserved.
 * Author: Ronan Collobert <locronan@fb.com>
 *
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

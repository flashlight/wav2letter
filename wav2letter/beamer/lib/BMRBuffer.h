/* (c) Ronan Collobert 2016, Facebook */

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

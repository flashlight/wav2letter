/*
 * Copyright 2017 Facebook
 * @author Vitaliy Liptchinsky (vitaliy888@fb.com)
 */

#pragma once

typedef struct WavState WavEvalState;

enum waveval_errorcodes {
  NO_ERROR = 0,
  BAD_ALLOC = 1,
  LUA_ERROR = 2,
};

WavEvalState* waveval_init(const char* modelPath, int* error);
long waveval_getModelDepth(WavEvalState* state, int* error);
float* waveval_forward(WavEvalState* state,
                          const float* input, long inputSize,
                          long depth,
                          long* outputSize, int* error);
void waveval_dispose(WavEvalState* state);
const char* waveval_geterror(const WavEvalState* state, int error);

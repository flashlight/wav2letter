// Copyright 2017 Facebook. All rights reserved.
//
// Author: Vitaliy Liptchinsky (vitaliy888@fb.com)

#include "../waveval.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  int error = 0;
  // models are too big to check in to repository, use your local conv model
  WavEvalState* state = waveval_init("model.bin",
                                           &error);
  if (error != NO_ERROR) {
    printf("Initialization failed: %s", waveval_geterror(state, error));
  }
  printf("Init succeeded!\n");

  long outputSize = 0;
  const long inputSize = 15000;
  float input[inputSize];
  for (int i = 0; i < 10; ++i) {
    long depth = waveval_getModelDepth(state, &error);
    if (error != NO_ERROR) {
      printf("LUA ERROR occured: ");
      printf("%s\n", waveval_geterror(state, error));
    }
    printf("getDepth succeeded. Depth: %ld!\n", depth);
    assert(depth == 27);

    float* output = waveval_forward(state, input, inputSize, 27, &outputSize,
                                       &error);
    if (error != NO_ERROR) {
      printf("LUA ERROR occured: ");
      printf("%s\n", waveval_geterror(state, error));
    }
    printf("feedForward succeeded!\n");

    assert(outputSize > 1);
    assert(output != NULL);
  }
  printf("Test succeeded!!!!\n");
  return 0;
}

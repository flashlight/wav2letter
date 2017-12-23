/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "waveval.h"
#include <assert.h>
#include <lua.h>
#include <luaT.h>
#include "TH.h"
#include "lualib.h"

const int c_moduleIndex = 1;
const int c_networkIndex = 2;
const int c_transformsIndex = 3;

// internal state
struct WavState {
  lua_State* L;
};


#define LUA_CHECK(hr, errcode)   \
  if ((hr)) {                    \
    *error = errcode;            \
    return result;               \
  }                              \

void waveval_dispose(WavEvalState* state) {
  if (state) {
    if (state->L) {
      lua_close(state->L);
    }
    free(state);
  }
}

WavEvalState* waveval_init(const char* modelPath, int* error) {
  assert(modelPath);
  assert(error);
  *error = NO_ERROR;
  WavEvalState* result = malloc(sizeof(WavEvalState));
  LUA_CHECK(!result, BAD_ALLOC);

  result->L = luaL_newstate();
  assert(result->L);

  luaL_openlibs(result->L);

  lua_getglobal(result->L, "require");
  lua_pushstring(result->L, "wav2letter.runtime.eval");
  LUA_CHECK(lua_pcall(result->L, 1, 1, 0), LUA_ERROR);
  assert(lua_gettop(result->L) == 1 && lua_istable(result->L, c_moduleIndex));

  lua_getfield(result->L, c_moduleIndex, "loadModel");
  assert(lua_isfunction(result->L, -1));
  lua_pushstring(result->L, modelPath);
  LUA_CHECK(lua_pcall(result->L, 1, 2, 0), LUA_ERROR);

  assert(lua_gettop(result->L) == c_transformsIndex); //invariant
  return result;
}

long waveval_getModelDepth(WavEvalState* state, int* error) {
  assert(state);
  assert(error);
  *error = NO_ERROR;
  long result = 0;

  //discard any leftover state from feedforward
  lua_settop(state->L, c_transformsIndex);
  lua_getfield(state->L, c_moduleIndex, "getModelDepth");
  assert(lua_isfunction(state->L, -1));
  lua_pushvalue(state->L, c_networkIndex); //copy network as an input param
  LUA_CHECK(lua_pcall(state->L, 1, 1, 0), LUA_ERROR);

  result = (size_t)lua_tonumber(state->L, -1);
  lua_pop(state->L, 1);
  assert(lua_gettop(state->L) == c_transformsIndex); //invariant
  return result;
}

float* waveval_forward(WavEvalState* state,
                       const float* input, long inputSize,
                       long depth,
                       long* outputSize, int* error) {
  assert(state);
  assert(error);
  float* result = NULL;
  *error = NO_ERROR;

  //discard any leftover state from feedforward
  lua_settop(state->L, c_transformsIndex);

  THFloatTensor* inputTensor = THFloatTensor_newWithSize1d(inputSize);
  LUA_CHECK(!inputTensor, BAD_ALLOC);
  memcpy(THFloatTensor_data(inputTensor), input, sizeof(float) * inputSize);

  lua_getfield(state->L, c_moduleIndex, "feedForward");
  lua_pushvalue(state->L, c_networkIndex);
  lua_pushvalue(state->L, c_transformsIndex);
  luaT_pushudata(state->L, inputTensor, "torch.FloatTensor");
  lua_pushnumber(state->L, depth);

  LUA_CHECK(lua_pcall(state->L, 4, 2, 0), LUA_ERROR);
  assert(lua_gettop(state->L) == c_transformsIndex + 2);
  THFloatTensor* outputTensor = luaT_toudata(state->L, -2, "torch.FloatTensor");
  assert(outputTensor);
  assert(THFloatTensor_isContiguous(outputTensor));
  *outputSize = (long)lua_tonumber(state->L, -1);
  result = THFloatTensor_data(outputTensor);
  return result;
}

const char* waveval_geterror(const WavEvalState* state, int error) {
  switch (error) {
    case BAD_ALLOC:
      return "Allocation failure!";
    case LUA_ERROR:
      return lua_tostring(state->L, -1);
    case NO_ERROR:
      return "No error detected!";
    default:
      return "Unknown error";
  }
}

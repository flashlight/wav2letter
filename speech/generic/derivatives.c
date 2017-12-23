/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/derivatives.c"
#else

static int speech_(Derivatives_forward)(lua_State *L)
{
  int dwindow = luaL_checkint(L, 1);
  int awindow = luaL_checkint(L, 2);
  THTensor *in   = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *out  = luaT_checkudata(L, 4, torch_Tensor);
  THArgCheck(THTensor_(nDimension)(in) == 2, 2,
             "Input tensor is expected to be 2D");
  THArgCheck(THTensor_(nDimension)(out) == 2, 3,
             "Output tensor is expected to be 2D");

  in  = THTensor_(newContiguous)(in);
  int n    = THTensor_(size)(in, 0);
  int clen = THTensor_(size)(in, 1);
  real *out_data = THTensor_(data)(out);
  real *in_data  = THTensor_(data)(in);
  int outclen = clen * 3;

  int i;
  int j;
  int t;
  int left_t;
  int right_t;
  real norm_factor;
  for(t=0; t<n; t++)
  {
    //Copy
    for(i=0; i<clen; i++)
    {
      out_data[t*outclen + i] = in_data[t*clen + i];
    }
    //Derivatives (appending)
    for(i=0; i<clen; i++)
    {
      out_data[t*outclen + i + clen] = 0;
      norm_factor = 0;
      for(j=1; j<=dwindow; j++)
      {
        left_t  = (t-j < 0) ? 0 : t-j;
        right_t = (t+j >= n) ? n-1 : t+j;
        out_data[t*outclen + i + clen] += j*(in_data[right_t*clen + i] -
                                          in_data[left_t*clen + i]);
        norm_factor += ((real) 2*j*j);
      }
      if (norm_factor > 0)
      {
        out_data[t*outclen + i + clen] /= norm_factor;
      }
    }
  }
  for(t=0; t<n; t++)
  {
    //Acceleration (appending)
    for(i=0; i<clen; i++)
    {
      out_data[t*outclen + i + 2*clen] = 0;
      norm_factor = 0;
      for(j=1; j<=dwindow; j++)
      {
        left_t  = (t-j < 0) ? 0 : t-j;
        right_t = (t+j >= n) ? n-1 : t+j;
        out_data[t*outclen + i + 2*clen] +=
                            j*(out_data[right_t*outclen + clen + i] -
                               out_data[left_t*outclen + clen + i]);
        norm_factor += ((real) 2*j*j);
      }
      if (norm_factor > 0)
      {
        out_data[t*outclen + i + 2*clen] /= norm_factor;
      }
    }
  }

  /* clean up */
  THTensor_(free)(in);
  return 1;
}

static const struct luaL_Reg speech_(Derivatives__) [] = {
  {"Derivatives_forward", speech_(Derivatives_forward)},
  {NULL, NULL}
};

static void speech_(Derivatives_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, speech_(Derivatives__), "speech");
  lua_pop(L,1);
}

#endif

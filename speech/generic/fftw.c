#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/fftw.c"
#else

static int speech_(Fftw_forward)(lua_State *L)
{
  THTensor *in   = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *out  = luaT_checkudata(L, 2, torch_Tensor);
  THArgCheck(THTensor_(nDimension)(in) == 1, 1,
             "Input tensor is expected to be 1D");
  THArgCheck(THTensor_(nDimension)(out) == 2, 2,
             "Output tensor is expected to be 2D");

  in  = THTensor_(newContiguous)(in);
  int n = THTensor_(size)(in, 0);
  real *out_data = THTensor_(data)(out);
  real *in_data = THTensor_(data)(in);

#if defined(TH_REAL_IS_DOUBLE)
  fftw_complex *out_complex = (fftw_complex*)out_data;
  double *in_real  = (double*)in_data;
  pthread_mutex_lock(&speech_fftw_plan_mutex);
  fftw_plan p = fftw_plan_dft_r2c_1d(n, in_real, out_complex, FFTW_ESTIMATE);
  pthread_mutex_unlock(&speech_fftw_plan_mutex);
  fftw_execute(p);
  pthread_mutex_lock(&speech_fftw_plan_mutex);
  fftw_destroy_plan(p);
  pthread_mutex_unlock(&speech_fftw_plan_mutex);
#else
  fftwf_complex *out_complex = (fftwf_complex*)out_data;
  float *in_real  = (float*)in_data;
  pthread_mutex_lock(&speech_fftw_plan_mutex);
  fftwf_plan p = fftwf_plan_dft_r2c_1d(n, in_real, out_complex, FFTW_ESTIMATE);
  pthread_mutex_unlock(&speech_fftw_plan_mutex);
  fftwf_execute(p);
  pthread_mutex_lock(&speech_fftw_plan_mutex);
  fftwf_destroy_plan(p);
  pthread_mutex_unlock(&speech_fftw_plan_mutex);
#endif

  /* Copy stuff to the redundant part */
  int i;
  for(i=n/2+1; i<n; i++)
  {
    out_data[2*i] = out_data[2*n-2*i];
    out_data[2*i+1] = -out_data[2*n-2*i+1];
  }

  /* clean up */
  lua_pushvalue(L, 2);
  THTensor_(free)(in);
  return 1;
}

static const struct luaL_Reg speech_(Fftw__) [] = {
  {"Fftw_forward", speech_(Fftw_forward)},
  {NULL, NULL}
};

static void speech_(Fftw_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, speech_(Fftw__), "speech");
  lua_pop(L,1);
}

#endif

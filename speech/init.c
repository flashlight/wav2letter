#include "TH.h"
#include "luaT.h"
#include "fftw3.h"
#include <pthread.h>

/* making fftw plan is not thread safe */
/* executing a plan is thread safe */
static pthread_mutex_t speech_fftw_plan_mutex = PTHREAD_MUTEX_INITIALIZER;

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define speech_(NAME) TH_CONCAT_3(speech_,Real,NAME)

#include "generic/fftw.c"
#include "THGenerateFloatTypes.h"

#include "generic/derivatives.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libspeech(lua_State *L);

int luaopen_libspeech(lua_State *L)
{
  lua_newtable(L);

  speech_FloatFftw_init(L);
  speech_FloatDerivatives_init(L);

  speech_DoubleFftw_init(L);
  speech_DoubleDerivatives_init(L);

  return 1;
}

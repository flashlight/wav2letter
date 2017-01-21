#include "TH.h"

float fccviterbi(
  THFloatTensor *acc,
  THLongTensor *macc,
  THFloatTensor *accp,
  THLongTensor *path,
  THFloatTensor *inp,
  THFloatTensor *trans,
  int T,
  int N)
{
  float *acc_t    = THFloatTensor_data(acc);
  float *acc_tp   = THFloatTensor_data(accp);
  long *macc_p    = THLongTensor_data(macc);
  long *path_p    = THLongTensor_data(path);
  float *inp_p    = THFloatTensor_data(inp);
  float *trans_p  = THFloatTensor_data(trans);

  memcpy(acc_t, inp_p, sizeof(float)*N);

  for (int t = 1; t < T; t++)
  {
    {
      memcpy(acc_tp, acc_t, sizeof(float)*N);
    }

    long *macc_t  = macc_p + (t)*N;
    float *inp_t  = inp_p + (t)*N;

    for (int i = 0; i < N; i++)
    {
      float *trans_i = trans_p + i*N;
      acc_t[i] = acc_tp[0] + trans_i[0];
      macc_t[i] = 0;
      //This cannot be parallelized in its current form!
      for (int j = 1; j < N; j++)
      {
        float s = acc_tp[j] + trans_i[j];
        if (s > acc_t[i])
        {
          macc_t[i] = j;
          acc_t[i] = s;
        }
      }
      acc_t[i]  += inp_t[i];
    }

  }

  long mi = 0;
  for (int j = 1; j < N; j++)
    if (acc_t[j] > acc_t[mi])
      mi = j;
  float s = acc_t[mi];

  path_p[(T-1)] = mi + 1;
  for (int t = T-2; t >= 0; t--)
  {
    long *macc_t  = macc_p + (t+1)*N;
    mi = macc_t[mi];
    path_p[t] = mi + 1; //Torch counts from 1
  }

  return s;
}

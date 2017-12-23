/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


#include "TH.h"
#include <stdint.h>
#include <float.h>

float asgfwfal(const float *inp_p, const long *target_p,
               const float *trans_p, double *falacc_p,
               long *macc_p, long T, long TN, long N)
{
  //fal fw
  falacc_p[0] = inp_p[target_p[0] - 1];

  double s1i[TN];
  double s2i[TN];
  double m_p[TN];
  double max[TN];

  for (long i = 0; i < TN; i++)
  {
    s1i[i] = trans_p[N*(target_p[i]-1) + target_p[i]-1];
    s2i[i] = (i > 0 ? trans_p[N*(target_p[i]-1) + target_p[i-1]-1] : 0);
  }

  for (long t = 1; t < T; t++)
  {
    double *falacc_tp = falacc_p + (t-1)*TN;
    double *falacc_t  = falacc_p + (t)*TN;
    const float *inp_t  = inp_p + (t)*N;
    long high = t < TN ? t : TN;
    long low = T - t < TN ? TN - (T - t) : 1;
    if (T - t >= TN)
    {
      falacc_t[0] = s1i[0] + falacc_tp[0] + inp_t[target_p[0]-1];
      falacc_t[0] = (float) falacc_t[0];
    }
    for (long i = low; i < high; i++)
    {
      double s1 = s1i[i] + falacc_tp[i];
      double s2 = s2i[i] + falacc_tp[i-1];
      if (s1 > s2)
      {
        max[i] = s1;
        m_p[i] = s2 - s1;
      }
      else
      {
        max[i] = s2;
        m_p[i] = s1 - s2;
      }
    }
    for (long i = low; i < high; i++)
      m_p[i] = log(1 + exp(m_p[i]));
    for (long i = low; i < high; i++)
    {
      falacc_t[i] = max[i] + m_p[i] + inp_t[target_p[i]-1];
      falacc_t[i] = (float) falacc_t[i];
    }
    if (high < TN)
    {
      falacc_t[high] = s2i[high] + falacc_tp[high-1] + inp_t[target_p[high] - 1];
      falacc_t[high] = (float) falacc_t[high];
    }
  }

  float res = falacc_p[T*TN-1];
  return res;
}

#define expm(x)  exp(-(x))

void asgbwfal(float *gem_p, const long *target_p,
              const double *falacc_p, double *falgacc_p,
              const float *trans_p, float *gtrans_p,
              long T, long TN, long N,
              float falscale)
{
  float s1i[TN];
  float s2i[TN];
  float *gtrans1_p[TN];
  float *gtrans2_p[TN];
  float subtrans[TN][2];
  for (long i = 0; i < TN; i++)
  {
    subtrans[i][0] = 0;
    subtrans[i][1] = 0;
  }

  for (long i = 0; i < TN; i++)
  {
    s1i[i] = trans_p[N*(target_p[i]-1) + target_p[i]-1];
    s2i[i] = (i > 0 ? trans_p[N*(target_p[i]-1) + target_p[i-1]-1] : 0);
    gtrans1_p[i] = gtrans_p + ((target_p[i]-1)*N + (target_p[i]-1));
    gtrans2_p[i] = (i > 0 ? gtrans_p + ((target_p[i]-1)*N + (target_p[i-1]-1)) : 0);
  }
  //bw
  falgacc_p[T*TN-1] = 1;
  for (long t = T-1; t > 0; t--)
  {
    float *gem_t   = gem_p + (t)*N;
    const double *falacc_tp  = falacc_p + (t-1)*TN;
    double *falgacc_t  = falgacc_p + (t)*TN;
    double *falgacc_tp = falgacc_p + (t-1)*TN;
    long high = t < TN ? t + 1 : TN;
    long low = T - t < TN ? TN - (T - t) : 0;
    for (long i = low; i < high; i++)
    {
      gem_t[target_p[i]-1] += (float) (falscale * falgacc_t[i]);

      if ((high < TN || t == TN - 1) && i == high - 1 && i > 0)
      {
        falgacc_tp[i-1] += falgacc_t[i];
        subtrans[i][1] += ((float) falgacc_t[i] * falscale);
      }
      else if (i == 0)
      {
        falgacc_tp[i]   += falgacc_t[i];
        subtrans[i][0] += ((float) falgacc_t[i] * falscale);
      }
      else
      {
        double m_1 = s1i[i] + falacc_tp[i];
        double m_2 = s2i[i] + falacc_tp[i-1];

        double max = -DBL_MAX;
        max = m_1 > m_2 ? m_1 : m_2;

        m_1 = expm(max - s1i[i] - falacc_tp[i]);
        m_2 = expm(max - s2i[i] - falacc_tp[i-1]);

        double s = 0;
        s += m_1;
        s += m_2;
        double s1 = m_1 / s;
        double s2 = m_2 / s;

        //TODO: Next try writing out the gsubtransitions explicitly
        //and adding them to gtransitions after all is done.

        subtrans[i][0] += ((float) s1 * falgacc_t[i] * falscale);
        subtrans[i][1] += ((float) s2 * falgacc_t[i] * falscale);

        falgacc_tp[i]   += s1 * falgacc_t[i];
        falgacc_tp[i-1] += s2 * falgacc_t[i];
      }

    }
  }
  gem_p[target_p[0]-1] += (float) (falscale * falgacc_p[0]);
  for (long i = 0; i < TN; i++)
  {
    gtrans_p[((target_p[i]-1)*N + (target_p[i]-1))] += subtrans[i][0];
    if (i > 0)
    {
      gtrans_p[((target_p[i]-1)*N + (target_p[i-1]-1))] += subtrans[i][1];
    }
  }
}


double asgfwfcc(const float *inp_p, const float *trans_p, long *fccmacc_p, double *fccacc_p, long T, long N)
{
  //fcc fw
  for (long i = 0; i < N; i++)
    fccacc_p[i] = inp_p[i];

  double *fccacc_tp;
  double *fccacc_t = fccacc_p;
  long *fccmacc_t;
  for (long t = 1; t < T; t++)
  {
    fccacc_tp = fccacc_p + (t-1)*N;
    fccacc_t  = fccacc_p + (t)*N;
    fccmacc_t = fccmacc_p + (t)*N;
    const float *inp_t  = inp_p + (t)*N;
    for (long i = 0; i < N; i++)
    {
      double max = -DBL_MAX;
      for (long j = 0; j < N; j++)
      {
        double z = trans_p[i*N + j] + fccacc_tp[j];
        if (max < z)
        {
          max = z;
          fccmacc_t[i] = j;
        }
      }

      double sum = 0;
      for (long j = 0; j < N; j++)
      {
        sum += expm(max - trans_p[i*N + j] - fccacc_tp[j]);
      }

      sum = max + log(sum) + inp_t[i];
      fccacc_t[i] = (float) sum;
    }
  }

  double max = -DBL_MAX;
  for (long i = 0; i < N; i++)
    if (max < fccacc_t[i]) max = fccacc_t[i];
  double sum = 0;
  for (long i = 0; i < N; i++)
    sum += expm(max - fccacc_t[i]);
  sum = log(sum) + max;
  return (float) sum;
}

void asgbwfcc(float *gem_p, const long *fccmacc_p, const double *fccacc_p, double *fccgacc_p, const float *trans_p, float *gtrans_p, long T, long N, float fccscale)
{
  //bw step 1
  const double *fccacc_t   = fccacc_p  + (T-1)*N;
  double *fccgacc_t  = fccgacc_p + (T-1)*N;
  double fccloss = 0;

  float *gem_t  = gem_p + (T-1)*N;
  double max = -DBL_MAX;
  for (long j = 0; j < N; j++)
    if (max < fccacc_t[j]) max = fccacc_t[j];

  double sum = 0;
  for (long j = 0; j < N; j++)
    sum += expm(max - fccacc_t[j]);

  for (long j = 0; j < N; j++)
    fccgacc_t[j] = expm(max - fccacc_t[j]) / sum;

  for (long i = 0; i < N; i++)
  {
    gem_t[i] += (float) (fccgacc_t[i] * fccscale);
  }

  //bw
  for (long t = T-2; t >= 0; t--)
  {
    const double *fccacc_t   = fccacc_p + (t)*N;
    double *fccgacc_t  = fccgacc_p + (t)*N;
    double *fccgacc_tp = fccgacc_p + (t+1)*N;
    const long *fccmacc_t   = fccmacc_p  + (t+1)*N;
    float *gem_t  = gem_p + (t)*N;

    double m_m[N*N];
    for (long i = 0; i < N; i++)
    {
      double max = trans_p[i*N + fccmacc_t[i]] + fccacc_t[fccmacc_t[i]];
      double sum = 0;
      for (long j = 0; j < N; j++)
      {
        m_m[i*N + j] = expm(max - trans_p[i*N + j] - fccacc_t[j]);
        sum = sum + m_m[i*N + j];
      }
      for (long j = 0; j < N; j++)
      {
        m_m[i*N + j] = m_m[i*N + j] / sum;
      }
    }

    for (long i = 0; i < N; i++)
    {
      for (long j = 0; j < N; j++)
        fccgacc_t[i] += m_m[j*N + i] * fccgacc_tp[j];
    }

    for (long i = 0; i < N; i++)
    {
      for (long j = 0; j < N; j++)
      {
        gtrans_p[j*N + i] += (float) m_m[j*N + i] * fccgacc_tp[j] * fccscale;
      }
    }

    for (long i = 0; i < N; i++)
    {
      gem_t[i] += fccgacc_t[i] * fccscale;
    }
  }
}

double falfw(THFloatTensor *input,
              THLongTensor  *target,
              THFloatTensor *trans,
              THDoubleTensor *acc,
              THLongTensor *macc,
              long T,
              long N,
              long TN)
{

  float *inp_p      = THFloatTensor_data(input);
  long *target_p = THLongTensor_data(target);
  float *trans_p    = THFloatTensor_data(trans);
  double *acc_p  = THDoubleTensor_data(acc);
  long *macc_p  = THLongTensor_data(macc);

  return asgfwfal(inp_p, target_p, trans_p, acc_p, macc_p, T, TN, N);
}

void falbw(THFloatTensor *input,
              THLongTensor  *target,
              THFloatTensor *trans,
              THFloatTensor *gem,
              THFloatTensor *gtrans,
              THDoubleTensor *acc,
              THDoubleTensor *gacc,
              float scale,
              long T,
              long N,
              long TN)
{
  float *inp_p      = THFloatTensor_data(input);
  long *target_p = THLongTensor_data(target);
  float *trans_p    = THFloatTensor_data(trans);
  float *gem_p      = THFloatTensor_data(gem);
  float *gtrans_p   = THFloatTensor_data(gtrans);
  double *acc_p  = THDoubleTensor_data(acc);
  double *gacc_p = THDoubleTensor_data(gacc);

  asgbwfal(gem_p, target_p, acc_p, gacc_p, trans_p, gtrans_p, T, TN, N, scale);
}

double fccfw(THFloatTensor *input,
              THFloatTensor *trans,
              THLongTensor *macc,
              THDoubleTensor *acc,
              long T,
              long N)
{

  float *inp_p      = THFloatTensor_data(input);
  float *trans_p    = THFloatTensor_data(trans);
  long *macc_p  = THLongTensor_data(macc);
  double *acc_p  = THDoubleTensor_data(acc);

  return asgfwfcc(inp_p, trans_p, macc_p, acc_p, T, N);
}

void fccbw(THFloatTensor *input,
              THFloatTensor *trans,
              THFloatTensor *gem,
              THFloatTensor *gtrans,
              THLongTensor *macc,
              THDoubleTensor *acc,
              THDoubleTensor *gacc,
              float scale,
              long T,
              long N)
{
  float *inp_p      = THFloatTensor_data(input);
  float *trans_p    = THFloatTensor_data(trans);
  float *gem_p      = THFloatTensor_data(gem);
  float *gtrans_p   = THFloatTensor_data(gtrans);
  long *macc_p   = THLongTensor_data(macc);
  double *acc_p  = THDoubleTensor_data(acc);
  double *gacc_p = THDoubleTensor_data(gacc);

  asgbwfcc(gem_p, macc_p, acc_p, gacc_p, trans_p, gtrans_p, T, N, scale);
}

void asgbatchfw(THFloatTensor **input,
              THLongTensor  **target,
              THFloatTensor *trans,
              THDoubleTensor *falacc,
              THLongTensor *falmacc,
              THDoubleTensor *falgacc,
              THDoubleTensor *fccacc,
              THLongTensor *fccmacc,
              THDoubleTensor *fccgacc,
              THFloatTensor  *falscale,
              THFloatTensor  *fccscale,
              THLongTensor  *T,
              long N,
              THLongTensor  *TN,
              THFloatTensor  *loss,
              long B)
{

  float *trans_p    = THFloatTensor_data(trans);
  double *falacc_p  = THDoubleTensor_data(falacc);
  long *falmacc_p = THLongTensor_data(falmacc);
  double *falgacc_p = THDoubleTensor_data(falgacc);
  double *fccacc_p  = THDoubleTensor_data(fccacc);
  long *fccmacc_p = THLongTensor_data(fccmacc);
  double *fccgacc_p = THDoubleTensor_data(fccgacc);
  float *loss_p     = THFloatTensor_data(loss);

  float *falscale_p = THFloatTensor_data(falscale);
  float *fccscale_p = THFloatTensor_data(fccscale);

  long *T_p  = THLongTensor_data(T);
  long *TN_p = THLongTensor_data(TN);

  int32_t fccacc_offset = THDoubleTensor_size(fccacc, 1) * THDoubleTensor_size(fccacc, 2);
  int32_t fccmacc_offset = THLongTensor_size(fccmacc, 1) * THLongTensor_size(fccmacc, 2);
  int32_t falacc_offset = THDoubleTensor_size(falacc, 1) * THDoubleTensor_size(falacc, 2);
  int32_t falmacc_offset = THLongTensor_size(falmacc, 1) * THLongTensor_size(falmacc, 2);

  for (int i = 0; i < B; i++) loss_p[i] = 0;

  float losss[2*B];
  #pragma omp parallel for
  for (int32_t ii = 0; ii < 2*B; ii++)
  {
    int i = ii/2;
    double loss;
    float *inp_p      = THFloatTensor_data(input[i]);
    long *target_p = THLongTensor_data(target[i]);
    if (ii % 2 == 0)
    {
      losss[ii] = asgfwfcc(inp_p,
                         trans_p,
                         fccmacc_p + i*fccmacc_offset,
                         fccacc_p + i*fccacc_offset,
                         T_p[i], N) * fccscale_p[i];
    } else {
      losss[ii] = asgfwfal(inp_p,
                         target_p,
                         trans_p,
                         falacc_p + i*falacc_offset,
                         falmacc_p + i*falmacc_offset,
                         T_p[i], TN_p[i], N) * falscale_p[i];
    }
  }
  for (int32_t ii = 0; ii < 2*B; ii++)
  {
    int i = ii/2;
    loss_p[i] += losss[ii];
  }
}

void asgbatchbw(THFloatTensor **input,
              THLongTensor  **target,
              THFloatTensor *trans,
              THFloatTensor *falgem,
              THFloatTensor *fccgem,
              THFloatTensor *falgtrans,
              THFloatTensor *fccgtrans,
              THDoubleTensor *falacc,
              THLongTensor *falmacc,
              THDoubleTensor *falgacc,
              THDoubleTensor *fccacc,
              THLongTensor *fccmacc,
              THDoubleTensor *fccgacc,
              THFloatTensor  *falscale,
              THFloatTensor  *fccscale,
              THLongTensor  *T,
              long N,
              THLongTensor  *TN,
              long B)
{
  float *trans_p    = THFloatTensor_data(trans);
  float *falgtrans_p   = THFloatTensor_data(falgtrans);
  float *fccgtrans_p   = THFloatTensor_data(fccgtrans);
  double *falacc_p  = THDoubleTensor_data(falacc);
  long *falmacc_p = THLongTensor_data(falmacc);
  double *falgacc_p = THDoubleTensor_data(falgacc);
  double *fccacc_p  = THDoubleTensor_data(fccacc);
  long *fccmacc_p = THLongTensor_data(fccmacc);
  double *fccgacc_p = THDoubleTensor_data(fccgacc);
  float *falgem_p      = THFloatTensor_data(falgem);
  float *fccgem_p      = THFloatTensor_data(fccgem);

  long *T_p  = THLongTensor_data(T);
  long *TN_p = THLongTensor_data(TN);

  float *falscale_p = THFloatTensor_data(falscale);
  float *fccscale_p = THFloatTensor_data(fccscale);

  int32_t gtrans_offset = N*N;
  int32_t gem_offset = THFloatTensor_size(falgem, 1) * THFloatTensor_size(fccgem, 2);
  int32_t fccacc_offset = THDoubleTensor_size(fccacc, 1) * THDoubleTensor_size(fccacc, 2);
  int32_t falacc_offset = THDoubleTensor_size(falacc, 1) * THDoubleTensor_size(falacc, 2);
  int32_t fccgacc_offset = THDoubleTensor_size(fccgacc, 1) * THDoubleTensor_size(fccgacc, 2);
  int32_t falgacc_offset = THDoubleTensor_size(falgacc, 1) * THDoubleTensor_size(falgacc, 2);
  int32_t fccmacc_offset = THLongTensor_size(fccmacc, 1) * THLongTensor_size(fccmacc, 2);
  int32_t falmacc_offset = THLongTensor_size(falmacc, 1) * THLongTensor_size(falmacc, 2);

  #pragma omp parallel for
  for (int32_t ii = 0; ii < 2*B; ii++)
  {
    int i = ii / 2;
    float *inp_p      = THFloatTensor_data(input[i]);
    long *target_p = THLongTensor_data(target[i]);
    if (ii % 2 == 0)
    {
      asgbwfcc(fccgem_p + i*gem_offset,
               fccmacc_p + i*fccacc_offset,
               fccacc_p + i*fccacc_offset,
               fccgacc_p + i*fccgacc_offset,
               trans_p,
               fccgtrans_p + i*gtrans_offset,
               T_p[i], N, fccscale_p[i]);
    } else {
      asgbwfal(falgem_p + i*gem_offset,
               target_p,
               falacc_p + i*falacc_offset,
               falgacc_p + i*falgacc_offset,
               trans_p,
               falgtrans_p + i*gtrans_offset,
               T_p[i], TN_p[i], N, falscale_p[i]);
    }
  }
}

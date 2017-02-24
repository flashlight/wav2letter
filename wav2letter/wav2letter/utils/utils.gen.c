void THTensor_(reduceMostFrequentIndex)(THTensor *dst, THTensor *src, int dimension, int N)
{
  THLongStorage *dim;
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(src), 2, "invalid dimension");
  dim = THTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(dst, dim, NULL);
  THLongStorage_free(dim);

  long *counts = THAlloc(sizeof(long)*N);
  TH_TENSOR_DIM_APPLY2(real, dst, real, src, dimension,
                       for(long n = 0; n < N; n++)
                         counts[n] = 0;
                       for(long i = 0; i < src_size; i++) {
                         real z = src_data[i*src_stride];
                         if(z < 1 || z > N)
                           THError("source index out of bound");
                         counts[z-1]++;
                       }
                       long maxval = 0;
                       long maxidx = 0;
                       for(long n = 0; n < N; n++) {
                         long z = counts[n];
                         if(z > maxval) {
                           maxval = z;
                           maxidx = n;
                         }
                       }
                       dst_data[0] = maxidx+1;
    )
  THFree(counts);
}

void THTensor_(uniq)(THTensor *dst, THTensor *src)
{
  THArgCheck(THTensor_(nDimension)(src) == 1, 1, "src should have only one dimension (be a vector)");
  long n = src->size[0];
  THTensor_(resize1d)(dst, n);
  real *dst_p = THTensor_(data)(dst);
  real *src_p = THTensor_(data)(src);

  dst_p[0] = src_p[0];
  long m = 1;
  for(long i = 1; i < n; i++) {
    if(src_p[i] != dst_p[m-1])
      dst_p[m++] = src_p[i];
  }

  THTensor_(resize1d)(dst, m);
}

void THTensor_(replabel)(THTensor *dst, THTensor *src, int replabel, THTensor *replabels)
{
  THArgCheck(THTensor_(nDimension)(src) == 1, 1, "src should have only one dimension (be a vector)");
  THArgCheck(THTensor_(nDimension)(replabels) == 1, 3, "replabels should be a 1d tensor");
  THArgCheck(THTensor_(size)(replabels, 0) == replabel, 3, "replabels should be of size <replabel>");

  long n = src->size[0];
  THTensor_(resize1d)(dst, n);
  real *dst_p = THTensor_(data)(dst);
  real *src_p = THTensor_(data)(src);
  real *replabels_p = THTensor_(data)(replabels);

  dst_p[0] = src_p[0];
  long m = 1;
  long s = 0;
  for(long i = 1; i < n; i++) {
    if(src_p[i] != dst_p[m-1])
    {
      if (s > 0)
        dst_p[m++] = replabels_p[s - 1];
      s = 0;
      dst_p[m++] = src_p[i];
    }
    else
      if (s < replabel)
        s += 1;
  }
  if (s > 0)
    dst_p[m++] = replabels_p[s - 1];

  THTensor_(resize1d)(dst, m);
}

void THTensor_(invreplabel)(THTensor *dst, THTensor *src, int replabel, THTensor *replabels)
{
  THArgCheck(THTensor_(nDimension)(src) == 1, 1, "src should have only one dimension (be a vector)");
  THArgCheck(THTensor_(nDimension)(replabels) == 1, 3, "replabels should be a 1d tensor");
  THArgCheck(THTensor_(size)(replabels, 0) == replabel, 3, "replabels should be of size <replabel>");
  long n = src->size[0];
  THTensor_(resize1d)(dst, (replabel + 1)*n);
  real *dst_p = THTensor_(data)(dst);
  real *src_p = THTensor_(data)(src);
  real *replabels_p = THTensor_(data)(replabels);
  long m = 0;
  dst_p[m++] = src_p[0];
  for(long i = 1; i < n; i++)
  {
    real src_i = src_p[i];
    long s = 0;
    for(long j = 0; j < replabel; j++) {
      if(src_i == replabels_p[j]) {
        s = j+1;
        break;
      }
    }
    if(s > 0)
    {
      for(long j = 0; j < s; j++)
        dst_p[m++] = src_p[i-1];
    }
    else
    {
      dst_p[m++] = src_p[i];
    }
  }

  THTensor_(resize1d)(dst, m);
}

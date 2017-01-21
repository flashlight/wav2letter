long THTensor_(editdistance)(THTensor *s, THTensor *t)
{
  THArgCheck(s->nDimension <= 1, 0, "vector (1D) expected");
  THArgCheck(t->nDimension <= 1, 0, "vector (1D) expected");
  long m = (s->nDimension > 0 ? s->size[0] : 0);
  long n = (t->nDimension > 0 ? t->size[0] : 0);
  long st_s = (s->nDimension > 0 ? s->stride[0] : 0);
  long st_t = (t->nDimension > 0 ? t->stride[0] : 0);
  real *s_p = THTensor_(data)(s);
  real *t_p = THTensor_(data)(t);

  long dist;

  long **d = malloc((m + 1) * sizeof(long*));
  d[0] = malloc((m + 1) * (n + 1) * sizeof(long));
  for(size_t i = 1; i < (m + 1); i++)
      d[i] = d[0] + i * (n + 1);

  for(long i = 0; i <= m; i++)
    d[i][0] = i;
  for(long j = 0; j <= n; j++)
    d[0][j] = j;

  for(long j = 1; j <= n; j++) {
    for(long i = 1; i <= m; i++) {
      if(s_p[(i-1)*st_s] == t_p[(j-1)*st_t]) {
        d[i][j] = d[i-1][j-1];
      }
      else {
        long z = d[i-1][j]+1;
        z = (z < d[i][j-1]+1 ? z : d[i][j-1]+1);
        z = (z < d[i-1][j-1]+1 ? z : d[i-1][j-1]+1);
        d[i][j] = z;
      }
    }
  }

  /* for(long j = 0; j <= n; j++) { */
  /*   for(long i = 0; i <= m; i++) { */
  /*     printf("%ld ", d[i][j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  dist = d[m][n];

  free(d[0]);
  free(d);

  return dist;
}

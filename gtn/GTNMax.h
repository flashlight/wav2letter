/*
 * (c) 2015 Facebook. All rights reserved.
 * Author: Ronan Collobert <locronan@fb.com>
 *
 */

#ifndef GTN_MAX_INC
#define GTN_MAX_INC

#include "GTN.h"

real GTNGraph_forward_max(GTNGraph* gtn, long maxidx, long *path, long *pathsize);
void GTNGraph_backward_max(GTNGraph *gtn, real g, long maxidx);

#endif

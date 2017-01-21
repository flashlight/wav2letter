/*
 * (c) 2015 Facebook. All rights reserved.
 * Author: Ronan Collobert <locronan@fb.com>
 *
 */

#ifndef GTN_LOGADD_INC
#define GTN_LOGADD_INC

#include "GTN.h"

real GTNGraph_forward_logadd(GTNGraph* gtn, long maxidx);
void GTNGraph_backward_logadd(GTNGraph *gtn, real g, long maxidx);

#endif

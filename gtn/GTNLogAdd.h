/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef GTN_LOGADD_INC
#define GTN_LOGADD_INC

#include "GTN.h"

real GTNGraph_forward_logadd(GTNGraph* gtn, long maxidx);
void GTNGraph_backward_logadd(GTNGraph *gtn, real g, long maxidx);

#endif

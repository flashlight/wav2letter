/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "inference/module/nn/Conv1d.h"
#include "inference/module/nn/Identity.h"
#include "inference/module/nn/LayerNorm.h"
#include "inference/module/nn/Linear.h"
#include "inference/module/nn/LocalNorm.h"
#include "inference/module/nn/Relu.h"
#include "inference/module/nn/Residual.h"
#include "inference/module/nn/Sequential.h"
#include "inference/module/nn/TDSBlock.h"

// We need to include the backend for the Cereal serirlization implementation.
#if W2L_INFERENCE_BACKEND == fbgemm
#include "inference/module/nn/backend/fbgemm/fbgemm.h"
#endif

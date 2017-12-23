/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef BMR_DECODER_INC
#define BMR_DECODER_INC

#include "BMRTrie.h"
#include "BMRLM.h"

typedef struct BMRDecoder_  BMRDecoder;

typedef struct BMRDecoderOptions_ {
  long beamsize; /* max beam size */
  float beamscore; /* max delta score kept in the beam */
  float lmweight; /* weight of lm */
  float wordscore; /* score for inserting a word */
  float unkscore; /* score for inserting an unknown */
  int forceendsil; /* force ending in a sil? */
  int logadd; /* use logadd instead of max when merging same word hypothesis */
  float silweight; /* silence is golden */
} BMRDecoderOptions;

BMRDecoder* BMRDecoder_new(BMRTrie *lexicon, BMRLM *lm, long sil, BMRTrieLabel unk);
void BMRDecoder_decode(BMRDecoder *decoder, BMRDecoderOptions *opt, float *transitions, float *emissions, long T, long N, long *nhyp_, float *scores_, long *llabels_, long *labels_);
long BMRDecoder_mem(BMRDecoder *decoder);
void BMRDecoder_free(BMRDecoder *decoder);
void BMRDecoder_settoword(BMRDecoder *decoder, const char* (*toword)(long)); /* for debugging purposes */
/* nhyp_, scores_, llabels_, labels_ may be NULL (in which case they are not returned) */
void BMRDecoder_store_hypothesis(BMRDecoder *decoder, long *nhyp_, float *scores_, long *llabels_, long *labels_);

#endif

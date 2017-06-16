/* (c) Ronan Collobert 2016, Facebook */

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
} BMRDecoderOptions;

BMRDecoder* BMRDecoder_new(BMRTrie *lexicon, BMRLM *lm, long sil, BMRTrieLabel unk);
void BMRDecoder_decode(BMRDecoder *decoder, BMRDecoderOptions *opt, float *transitions, float *emissions, long T, long N, long *nhyp_, float *scores_, long *llabels_, long *labels_);
long BMRDecoder_mem(BMRDecoder *decoder);
void BMRDecoder_free(BMRDecoder *decoder);
void BMRDecoder_settoword(BMRDecoder *decoder, const char* (*toword)(long)); /* for debugging purposes */

#endif

/* (c) Ronan Collobert 2016, Facebook */

#include "BMRDecoder.h"
#include "BMRBuffer.h"
#include "BMRArray.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h> /* DEBUG: */

typedef struct BMRDecoderNode_ {
  BMRLMState *lmstate; /* language model state */
  BMRTrieNode *lex; /* position in lexicon */
  struct BMRDecoderNode_ *parent; /* to backtrack */
  float score; /* score so far */
  float wordscore; /* word score so far (TODO) */
  long label; /* label of word (-1 if nothing) */
  /* BMRList *merged; /\* chained list of merged nodes, if any (TODO) *\/ */
  int alive; /* (TODO) */
} BMRDecoderNode;

struct BMRDecoder_ {
  const char* (*toword)(long);
  BMRTrie *lexicon;
  BMRLM *lm;
  BMRBuffer *candidates; /* store candidates nodes at current frame */
  BMRBuffer *nodes; /* store selected nodes for all frames */
  float candidatesbestscore;
  long sil; /* silence label (letter) */
  long unk; /* unknown label (word) */
};

void BMRDecoder_settoword(BMRDecoder *decoder, const char* (*toword)(long))
{
  decoder->toword = toword;
}

/* logadd */
#define MINUS_LOG_THRESHOLD -39.14
static double BMRLogAdd(double log_a, double log_b)
{
  double minusdif;
  if (log_a < log_b)
  {
    double tmp = log_a;
    log_a = log_b;
    log_b = tmp;
  }
  minusdif = log_b - log_a;
  if (minusdif < MINUS_LOG_THRESHOLD)
    return log_a;
  else
    return log_a + log1p(exp(minusdif));
}

static BMRDecoderNode* BMRDecoder_node_new(
  BMRBuffer *buffer,
  BMRLMState *lmstate,
  BMRTrieNode *lex,
  BMRDecoderNode *parent,
  float score,
  float wordscore,
  long label)
{
  BMRDecoderNode *node = BMRBuffer_grow(buffer, BMRBuffer_size(buffer)+1);
  if(node) {
    node->lmstate = lmstate;
    node->lex = lex;
    node->parent = parent;
    node->score = score;
    node->wordscore = wordscore;
    node->label = label;
    /* node->merged = NULL; */
    node->alive = 0;
  }
  return node;
}

static BMRDecoderNode *BMRDecoder_node_clone(
  BMRBuffer *buffer,
  BMRDecoderNode *node)
{
  BMRDecoderNode *newnode = BMRBuffer_grow(buffer, BMRBuffer_size(buffer)+1);
  memcpy(newnode, node, sizeof(BMRDecoderNode));
  return newnode;
}

static int BMRDecoder_candidates_reset(BMRDecoder *decoder)
{
  BMRBuffer_reset(decoder->candidates);
  decoder->candidatesbestscore = -INFINITY;
  return 0;
}

static int BMRDecoder_candidates_add(
  BMRDecoder *decoder,
  BMRDecoderOptions *opt,
  BMRLMState *lmstate,
  BMRTrieNode *lex,
  BMRDecoderNode *parent,
  float score,
  float wordscore,
  long label)
{
  if(score > decoder->candidatesbestscore) {
    decoder->candidatesbestscore = score;
  }

  if(score >= decoder->candidatesbestscore-opt->beamscore) { /* keep? */
    if(!BMRDecoder_node_new(decoder->candidates, lmstate, lex, parent, score, wordscore, label))
      return 1;
  }

  return 0;
}

static int BMRDecoder_compare_nodes_shortlist(const void *node1_, const void *node2_)
{
  BMRDecoderNode *node1 = *(BMRDecoderNode**)node1_;
  BMRDecoderNode *node2 = *(BMRDecoderNode**)node2_;
  int lmcmp = BMRLMState_compare(node1->lmstate, node2->lmstate);
  if(lmcmp != 0) {
    return lmcmp;
  }
  else { /* same lmstate */
    if(node1->lex < node2->lex) {
      return 1;
    }
    else if(node1->lex > node2->lex) {
      return -1;
    }
    else { /* same lmstate, same lex */

      if(node1->score < node2->score) {
        return 1;
      }
      else if(node1->score > node2->score) {
        return -1;
      }
      return 0;

    }
  }
}

static int BMRDecoder_compare_nodes(const void *node1_, const void *node2_)
{
  BMRDecoderNode *node1 = *(BMRDecoderNode**)node1_;
  BMRDecoderNode *node2 = *(BMRDecoderNode**)node2_;
  if(node1->score < node2->score) {
    return 1;
  }
  else if(node1->score > node2->score) {
    return -1;
  }
  return 0;
}

static int BMRDecoder_candidates_store(BMRDecoder *decoder, BMRDecoderOptions *opt, BMRArray *candidates, int issort)
{
  if(BMRBuffer_size(decoder->candidates) == 0)
    return 0;

  for(long i = 0; i < BMRBuffer_size(decoder->candidates); i++) {
    BMRDecoderNode *candidate = BMRBuffer_idx(decoder->candidates, i);
    if(candidate->score >= decoder->candidatesbestscore-opt->beamscore) { /* keep? */
      BMRArray_add(candidates, candidate);
    }
  }

  /* sort by (lmstate, lex, score) */
  BMRArray_sort(candidates, BMRDecoder_compare_nodes_shortlist);

  /* merge identical (lmstate, lex) */
#define CANDIDATE(idx) ((BMRDecoderNode*)BMRArray_get(candidates, (idx)))
  long m = 1;
  if(opt->logadd) { /* logadd version */
    double score = CANDIDATE(0)->score;
    for(long i = 1; i < BMRArray_size(candidates); i++) {
      if(
        BMRLMState_compare(CANDIDATE(i)->lmstate, CANDIDATE(m-1)->lmstate) ||
        CANDIDATE(i)->lex != CANDIDATE(m-1)->lex) {
        CANDIDATE(m-1)->score = score;
        BMRArray_set(candidates, m, CANDIDATE(i));
        score = CANDIDATE(m)->score;
        m++;
      }
      else {
        score = BMRLogAdd(score, CANDIDATE(i)->score);
      }
    }
    CANDIDATE(m-1)->score = score; /* do not forget final one */
  }
  else { /* max version */
    for(long i = 1; i < BMRArray_size(candidates); i++) {
      if(
        BMRLMState_compare(CANDIDATE(i)->lmstate, CANDIDATE(m-1)->lmstate) ||
        CANDIDATE(i)->lex != CANDIDATE(m-1)->lex) {
        BMRArray_set(candidates, m, CANDIDATE(i));
        m++;
      }
    }
  }
  BMRArray_resize(candidates, m);
#undef CANDIDATE

  /* sort remaining candidates by score (top-k) */
#if defined(BMR_USE_QUICKSELECT)
  if(m > opt->beamsize) {
    BMRArray_topksort(candidates, BMRDecoder_compare_nodes, opt->beamsize);
    BMRArray_resize(candidates, opt->beamsize);
  }
  /* needed only at the final step to return best hypothesis */
  if(issort)
    BMRArray_sort(candidates, BMRDecoder_compare_nodes);
#else
  BMRArray_sort(candidates, BMRDecoder_compare_nodes);
  if(BMRArray_size(candidates) > opt->beamsize)
    BMRArray_resize(candidates, opt->beamsize);
#endif

  /* store a copy of remaining nodes */
  for(long i = 0; i < BMRArray_size(candidates); i++) {
    BMRArray_set(
      candidates,
      i,
      BMRDecoder_node_clone(
        decoder->nodes,
        BMRArray_get(candidates, i))
      );
  }
  return 0;
}

BMRDecoder* BMRDecoder_new(BMRTrie *lexicon, BMRLM *lm, long sil, long unk)
{
  BMRDecoder *decoder = malloc(sizeof(BMRDecoder));
  if(decoder) {
    decoder->lexicon = lexicon; /* DEBUG: refcount? */
    decoder->lm = lm; /* DEBUG: refcount? */
    decoder->sil = sil;
    decoder->unk = unk;
    decoder->nodes = BMRBuffer_new(sizeof(BMRDecoderNode));
    decoder->candidates = BMRBuffer_new(sizeof(BMRDecoderNode));
    if(!decoder->nodes || !decoder->candidates) {
      BMRBuffer_free(decoder->nodes);
      BMRBuffer_free(decoder->candidates);
      free(decoder);
      return NULL;
    }
  }
  return decoder;
}

void BMRDecoder_decode(BMRDecoder *decoder, BMRDecoderOptions *opt, float *transitions, float *emissions, long T, long N, long *nhyp_, float *scores_, long *llabels_, long *labels_)
{
  long t, n;

  /* score, label, lex, parent */
  BMRArray **hyp = calloc(T+2, sizeof(BMRArray*)); /* count 0 and T+1 */
  for(long i = 0; i < T+2; i++)
    hyp[i] = BMRArray_new(0);

  if(!hyp ||
     (decoder->sil != BMRTrieNode_idx(BMRTrie_root(decoder->lexicon)))) {
    /* an error occured */
    *nhyp_ = -1;
    return;
  }

  BMRBuffer_reset(decoder->nodes); /* make sure we do not keep allocating nodes! */

  /* note: the lm reset itself with :start() */
  BMRArray_add(
    hyp[0],
    BMRDecoder_node_new(
      decoder->nodes,
      BMRLM_start(decoder->lm, 0),
      BMRTrie_root(decoder->lexicon),
      NULL,
      0, 0, -1)
    );

  for(t = 0; t < T; t++) {
    BMRDecoder_candidates_reset(decoder);
    for(n = 0; n < N; n++) {
      for(long h = 0; h < BMRArray_size(hyp[t]); h++) {
        BMRDecoderNode *prevhyp = BMRArray_get(hyp[t], h);
        long prevn = BMRTrieNode_idx(prevhyp->lex);
        float score = prevhyp->score + emissions[t*N+n] + (t > 0 ? transitions[n*N+prevn] : 0);
        BMRTrieNode *lex = prevhyp->lex;
        /* quantity which has been already taken in account (with smearing) at each hop in the lexicon */
        float lexmaxscore = (lex == BMRTrie_root(decoder->lexicon) ? 0 : BMRTrieNode_maxscore(lex));
        BMRLMState *lmstate = prevhyp->lmstate;

        /* emit a word only if silence */
        if(n == decoder->sil) {
          for(long i = 0; i < BMRTrieNode_nlabel(lex); i++) { /* true word? */
            float lmscore;
            BMRLMState *newlmstate = BMRLM_score(decoder->lm, lmstate, BMRTrieNode_label(lex, i), &lmscore);
            BMRDecoder_candidates_add(decoder, opt, newlmstate, BMRTrie_root(decoder->lexicon), prevhyp, score+opt->lmweight*(lmscore-lexmaxscore)+opt->wordscore, 0, BMRTrieNode_label(lex, i));
          }
          if((BMRTrieNode_nlabel(lex) == 0) && (opt->unkscore > -FLT_MAX)) { /* unknown? */
            float lmscore;
            BMRLMState *newlmstate = BMRLM_score(decoder->lm, lmstate, decoder->unk, &lmscore);
            BMRDecoder_candidates_add(decoder, opt, newlmstate, BMRTrie_root(decoder->lexicon), prevhyp, score+opt->lmweight*(lmscore-lexmaxscore)+opt->unkscore, 0, decoder->unk);
          }
          if(t == 0) { /* allow starting with a sil */
            BMRDecoder_candidates_add(decoder, opt, lmstate, lex, prevhyp, score, 0, -1);
          }
        }

        if(n == prevn && t > 0) { /* same place in lexicon (or sil) */
          BMRDecoder_candidates_add(decoder, opt, lmstate, lex, prevhyp, score, 0, -1);
        }
        else if(n != decoder->sil) { /* we assume sil cannot be in the lexicon */
          /* we eat-up a new token */
          lex = BMRTrieNode_child(lex, n);
          if(lex) { /* valid word(part) in lexicon? */
            /* may continue along current lex */
            BMRDecoder_candidates_add(decoder, opt, lmstate, lex, prevhyp, score+opt->lmweight*(BMRTrieNode_maxscore(lex)-lexmaxscore), 0, -1);

            /* DEBUG: works but emit word as soon as it find one valid */
            /* only for no-sil (NYI) */
            /* if(BMRTrieNode_label(lex) >= 0) { /\* true word? *\/ */
            /*   BMRDecoder_candidates_add(decoder, opt, score, BMRTrie_root(decoder->lexicon), BMRTrieNode_label(lex), prevhyp); */
            /* } */
          }
        }
      }
    }
    BMRDecoder_candidates_store(decoder, opt, hyp[t+1], 0); /* 0 was the start; no need to sort */
  }

  /* finish up */
  BMRDecoder_candidates_reset(decoder);
  for(long h = 0; h < BMRArray_size(hyp[T]); h++) {
    BMRDecoderNode *prevhyp = BMRArray_get(hyp[T], h);
    BMRTrieNode *lex = prevhyp->lex;
    float lexmaxscore = (lex == BMRTrie_root(decoder->lexicon) ? 0 : BMRTrieNode_maxscore(lex));
    BMRLMState *lmstate = prevhyp->lmstate;
    long prevn = BMRTrieNode_idx(prevhyp->lex);

    /* emit a word only if silence (... or here for end of sentence!!) */
    /* one could ignore this guy and force to finish in a sil (if sil is provided) */
    for(long i = 0; i < BMRTrieNode_nlabel(prevhyp->lex); i++) { /* true word? */
      float lmscore;
      float lmscoreend;
      BMRLMState *newlmstate = BMRLM_score(decoder->lm, lmstate, BMRTrieNode_label(lex, i), &lmscore);
      newlmstate = BMRLM_finish(decoder->lm, newlmstate, &lmscoreend);
      BMRDecoder_candidates_add(decoder, opt, newlmstate, BMRTrie_root(decoder->lexicon), prevhyp, prevhyp->score+opt->lmweight*(lmscore+lmscoreend-lexmaxscore)+opt->wordscore, 0, BMRTrieNode_label(lex, i));
    }

    /* we can also end in a sil */
    /* not enforcing that, we would end up in middle of a word */
    if((!opt->forceendsil) || (prevn == decoder->sil))
    {
      float lmscoreend;
      BMRLMState *newlmstate = BMRLM_finish(decoder->lm, lmstate, &lmscoreend);
      BMRDecoder_candidates_add(decoder, opt, newlmstate, lex, prevhyp, prevhyp->score+opt->lmweight*lmscoreend, 0, -1);
    }
  }
  BMRDecoder_candidates_store(decoder, opt, hyp[T+1], 1); /* sort */

  for(long r = 0; r < BMRArray_size(hyp[T+1]); r++) {
    BMRDecoderNode *node = BMRArray_get(hyp[T+1], r);
    scores_[r] = node->score;
    long i = 0;
    while(node) {
      labels_[r*(T+2)+T+1-i] = node->label;
      llabels_[r*(T+2)+T+1-i] = BMRTrieNode_idx(node->lex);
      node = node->parent;
      i++;
    }
  }
  *nhyp_ = BMRArray_size(hyp[T+1]);

  for(long i = 0; i < T+2; i++) {
    BMRArray_free(hyp[i]);
  }
  free(hyp);
}

long BMRDecoder_mem(BMRDecoder *decoder)
{
  return BMRBuffer_mem(decoder->nodes)+BMRBuffer_mem(decoder->candidates)+sizeof(BMRDecoder);
}

void BMRDecoder_free(BMRDecoder *decoder)
{
  if(decoder) {
    BMRBuffer_free(decoder->nodes);
    BMRBuffer_free(decoder->candidates);
    free(decoder);
  }
}

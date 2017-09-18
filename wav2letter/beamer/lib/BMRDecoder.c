/* (c) Ronan Collobert 2016, Facebook */

#include "BMRDecoder.h"
#include "BMRBuffer.h"
#include "BMRArray.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h> /* DEBUG: */

struct BMRDecoderNode_;
struct BMRDecoderNodeList_;

typedef struct BMRDecoderNodeList_ {
  struct BMRDecoderNode_ *node;
  struct BMRDecoderNodeList_ *next;
} BMRDecoderNodeList;

typedef struct BMRDecoderNode_ {
  BMRLMState *lmstate; /* language model state */
  BMRTrieNode *lex; /* position in lexicon */
  struct BMRDecoderNode_ *parent; /* to backtrack */
  float score; /* score so far */
  float gscore; /* gradient */
  float wordscore; /* word score so far (TODO) */
  BMRTrieLabel *label; /* label of word (-1 if nothing) */
  BMRDecoderNodeList *merged; /* chained list of merged nodes, if any */
  int active; /* (TODO) */
} BMRDecoderNode;

struct BMRDecoder_ {
  const char* (*toword)(long);
  BMRTrie *lexicon;
  BMRLM *lm;
  BMRBuffer *candidates; /* store candidates nodes at current frame */
  BMRBuffer *nodes; /* store selected nodes for all frames */
  BMRBuffer *merged; /* all merged list nodes */
  float candidatesbestscore;
  long sil; /* silence label (letter) */
  BMRTrieLabel unk; /* unknown label (word) */
  BMRArray *hyp; /* current hypothesis (Array (0..T+1) of Array (0..#hyp_t-1) */
};

void BMRDecoder_settoword(BMRDecoder *decoder, const char* (*toword)(long))
{
  decoder->toword = toword;
}

static double BMRDecoderNodeList_max(BMRDecoderNodeList *lst)
{
  double m = lst->node->score;
  while(lst->next) {
    lst = lst->next;
    if(lst->node->score > m) {
      m = lst->node->score;
    }
  }
  return m;
}

static void BMRDecoderNodeList_dmax(BMRDecoderNodeList *lst, double g)
{
  double m = lst->node->score;
  BMRDecoderNode *o = lst->node;
  lst->node->gscore = 0;
  while(lst->next) {
    lst = lst->next;
    lst->node->gscore = 0;
    if(lst->node->score > m) {
      m = lst->node->score;
      o = lst->node;
    }
  }
  o->gscore = g;
}

static double BMRDecoderNodeList_logadd(BMRDecoderNodeList *lst)
{
  double m = BMRDecoderNodeList_max(lst);
  double s = 0;
  while(lst) {
    s += exp(lst->node->score-m);
    lst = lst->next;
  }
  s = log(s) + m;
  return s;
}

static void BMRDecoderNodeList_dlogadd(BMRDecoderNodeList *lst, double g)
{
  double m = BMRDecoderNodeList_max(lst);
  double s = 0;
  BMRDecoderNodeList *lst_ = lst;
  while(lst) {
    lst->node->gscore = exp(lst->node->score-m);
    s += lst->node->gscore;
    lst = lst->next;
  }
  lst = lst_;
  while(lst) {
    lst->node->gscore *= (g/s);
    lst = lst->next;
  }
}

static BMRDecoderNode* BMRDecoder_node_new(
  BMRBuffer *buffer,
  BMRLMState *lmstate,
  BMRTrieNode *lex,
  BMRDecoderNode *parent,
  float score,
  float wordscore,
  BMRTrieLabel *label,
  BMRDecoderNodeList *merged)
{
  BMRDecoderNode *node = BMRBuffer_grow(buffer, BMRBuffer_size(buffer)+1);
  if(node) {
    node->lmstate = lmstate;
    node->lex = lex;
    node->parent = parent;
    node->score = score;
    node->gscore = 0;
    node->wordscore = wordscore;
    node->label = label;
    node->merged = merged;
    node->active = 0;
  }
  return node;
}

static BMRDecoderNodeList* BMRDecoder_node_list_new(
  BMRBuffer *buffer,
  BMRDecoderNode *node,
  BMRDecoderNodeList *next)
{
  BMRDecoderNodeList *lst = BMRBuffer_grow(buffer, BMRBuffer_size(buffer)+1);
  if(lst) {
    lst->node = node;
    lst->next = next;
  }
  return lst;
}

static BMRDecoderNode *BMRDecoder_node_clone(
  BMRBuffer *buffer,
  BMRDecoderNode *node)
{
  BMRDecoderNode *newnode = BMRBuffer_grow(buffer, BMRBuffer_size(buffer)+1);
  memcpy(newnode, node, sizeof(BMRDecoderNode));
  BMRDecoderNodeList *merged = node->merged;
  while(merged) {
    merged->node = BMRDecoder_node_clone(buffer, merged->node);
    merged = merged->next;
  }
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
  BMRTrieLabel *label)
{
  if(score > decoder->candidatesbestscore) {
    decoder->candidatesbestscore = score;
  }

  if(score >= decoder->candidatesbestscore-opt->beamscore) { /* keep? */
    if(!BMRDecoder_node_new(decoder->candidates, lmstate, lex, parent, score, wordscore, label, NULL))
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

static BMRDecoderNode* BMRDecoder_finalize_merge(BMRDecoder *decoder,  BMRDecoderOptions *opt, BMRDecoderNode *best, BMRDecoderNodeList *merged)
{
  merged = BMRDecoder_node_list_new(decoder->merged, best, merged);
  double score = (opt->logadd ? BMRDecoderNodeList_logadd(merged) : BMRDecoderNodeList_max(merged));
  return BMRDecoder_node_new(decoder->candidates, best->lmstate, best->lex, best->parent, score, best->wordscore, best->label, merged);
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
  BMRDecoderNodeList *merged = NULL;
  long m = 1;
  for(long i = 1; i < BMRArray_size(candidates); i++) {
    if(
      BMRLMState_compare(CANDIDATE(i)->lmstate, CANDIDATE(m-1)->lmstate) ||
      CANDIDATE(i)->lex != CANDIDATE(m-1)->lex) {
      if(merged) {
        BMRArray_set(candidates, m-1, BMRDecoder_finalize_merge(decoder, opt, CANDIDATE(m-1), merged));
        merged = NULL;
      }
      BMRArray_set(candidates, m, CANDIDATE(i));
      m++;
    }
    else {
      merged = BMRDecoder_node_list_new(decoder->merged, CANDIDATE(i), merged);
    }
  }
  /* do not forget final one */
  if(merged) {
    BMRArray_set(candidates, m-1, BMRDecoder_finalize_merge(decoder, opt, CANDIDATE(m-1), merged));
    merged = NULL;
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

BMRDecoder* BMRDecoder_new(BMRTrie *lexicon, BMRLM *lm, long sil, BMRTrieLabel unk)
{
  BMRDecoder *decoder = malloc(sizeof(BMRDecoder));
  if(decoder) {
    decoder->lexicon = lexicon; /* DEBUG: refcount? */
    decoder->lm = lm; /* DEBUG: refcount? */
    decoder->sil = sil;
    decoder->unk = unk;
    decoder->nodes = BMRBuffer_new(sizeof(BMRDecoderNode));
    decoder->candidates = BMRBuffer_new(sizeof(BMRDecoderNode));
    decoder->merged = BMRBuffer_new(sizeof(BMRDecoderNodeList));
    decoder->hyp = BMRArray_new(0);
    if(!decoder->nodes || !decoder->candidates || !decoder->merged || !decoder->hyp) {
      BMRBuffer_free(decoder->nodes);
      BMRBuffer_free(decoder->candidates);
      BMRBuffer_free(decoder->merged);
      BMRArray_free(decoder->hyp);
      free(decoder);
      return NULL;
    }
  }
  return decoder;
}

BMRArray* BMRDecoder_hyp_resize(BMRDecoder *decoder, long size)
{
  for(long i = 0; i < BMRArray_size(decoder->hyp); i++) {
    BMRArray_free(BMRArray_get(decoder->hyp, i));
  }
  BMRArray_resize(decoder->hyp, size);
  for(long i = 0; i < size; i++) {
    BMRArray_set(decoder->hyp, i, BMRArray_new(0));
  }
  return decoder->hyp;
}

void BMRDecoder_decode(BMRDecoder *decoder, BMRDecoderOptions *opt, float *transitions, float *emissions, long T, long N, long *nhyp_, float *scores_, long *llabels_, long *labels_)
{
  long t, n;

  /* score, label, lex, parent */
  BMRArray *hyp = BMRDecoder_hyp_resize(decoder, T+3); /* count 0 and T+1 and possible mergedscorenode */

  if(!hyp ||
     (decoder->sil != BMRTrieNode_idx(BMRTrie_root(decoder->lexicon)))) {
    /* an error occured */
    if(nhyp_) {
      *nhyp_ = -1;
    }
    return;
  }

  BMRBuffer_reset(decoder->nodes); /* make sure we do not keep allocating nodes! */
  BMRBuffer_reset(decoder->merged); /* make sure we do not keep allocating list nodes! */

  /* note: the lm reset itself with :start() */
  BMRArray_add(
    BMRArray_get(hyp, 0),
    BMRDecoder_node_new(
      decoder->nodes,
      BMRLM_start(decoder->lm, 0),
      BMRTrie_root(decoder->lexicon),
      NULL,
      0, 0, NULL, NULL)
    );

  for(t = 0; t < T; t++) {
    BMRDecoder_candidates_reset(decoder);
    for(n = 0; n < N; n++) {
      for(long h = 0; h < BMRArray_size(BMRArray_get(hyp, t)); h++) {
        BMRDecoderNode *prevhyp = BMRArray_get(BMRArray_get(hyp, t), h);
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
            BMRLMState *newlmstate = BMRLM_score(decoder->lm, lmstate, BMRTrieNode_label(lex, i)->lm, &lmscore);
            BMRDecoder_candidates_add(decoder, opt, newlmstate, BMRTrie_root(decoder->lexicon), prevhyp, score+opt->lmweight*(lmscore-lexmaxscore)+opt->wordscore, 0, BMRTrieNode_label(lex, i));
          }
          if((BMRTrieNode_nlabel(lex) == 0) && (opt->unkscore > -FLT_MAX)) { /* unknown? */
            float lmscore;
            BMRLMState *newlmstate = BMRLM_score(decoder->lm, lmstate, decoder->unk.lm, &lmscore);
            BMRDecoder_candidates_add(decoder, opt, newlmstate, BMRTrie_root(decoder->lexicon), prevhyp, score+opt->lmweight*(lmscore-lexmaxscore)+opt->unkscore, 0, &decoder->unk);
          }
          if(t == 0) { /* allow starting with a sil */
            BMRDecoder_candidates_add(decoder, opt, lmstate, lex, prevhyp, score, 0, NULL);
          }
        }

        if(n == prevn && t > 0) { /* same place in lexicon (or sil) */
          BMRDecoder_candidates_add(decoder, opt, lmstate, lex, prevhyp, score, 0, NULL);
        }
        else if(n != decoder->sil) { /* we assume sil cannot be in the lexicon */
          /* we eat-up a new token */
          lex = BMRTrieNode_child(lex, n);
          if(lex) { /* valid word(part) in lexicon? */
            /* may continue along current lex */
            BMRDecoder_candidates_add(decoder, opt, lmstate, lex, prevhyp, score+opt->lmweight*(BMRTrieNode_maxscore(lex)-lexmaxscore), 0, NULL);

            /* DEBUG: works but emit word as soon as it find one valid */
            /* only for no-sil (NYI) */
            /* if(BMRTrieNode_label(lex) >= 0) { /\* true word? *\/ */
            /*   BMRDecoder_candidates_add(decoder, opt, score, BMRTrie_root(decoder->lexicon), BMRTrieNode_label(lex), prevhyp); */
            /* } */
          }
        }
      }
    }
    BMRDecoder_candidates_store(decoder, opt, BMRArray_get(hyp, t+1), 0); /* 0 was the start; no need to sort */
  }

  /* finish up */
  BMRDecoder_candidates_reset(decoder);
  for(long h = 0; h < BMRArray_size(BMRArray_get(hyp, T)); h++) {
    BMRDecoderNode *prevhyp = BMRArray_get(BMRArray_get(hyp, T), h);
    BMRTrieNode *lex = prevhyp->lex;
    float lexmaxscore = (lex == BMRTrie_root(decoder->lexicon) ? 0 : BMRTrieNode_maxscore(lex));
    BMRLMState *lmstate = prevhyp->lmstate;
    long prevn = BMRTrieNode_idx(prevhyp->lex);

    /* emit a word only if silence (... or here for end of sentence!!) */
    /* one could ignore this guy and force to finish in a sil (if sil is provided) */
    for(long i = 0; i < BMRTrieNode_nlabel(prevhyp->lex); i++) { /* true word? */
      float lmscore;
      float lmscoreend;
      BMRLMState *newlmstate = BMRLM_score(decoder->lm, lmstate, BMRTrieNode_label(lex, i)->lm, &lmscore);
      newlmstate = BMRLM_finish(decoder->lm, newlmstate, &lmscoreend);
      BMRDecoder_candidates_add(decoder, opt, newlmstate, BMRTrie_root(decoder->lexicon), prevhyp, prevhyp->score+opt->lmweight*(lmscore+lmscoreend-lexmaxscore)+opt->wordscore, 0, BMRTrieNode_label(lex, i));
    }

    /* we can also end in a sil */
    /* not enforcing that, we would end up in middle of a word */
    if((!opt->forceendsil) || (prevn == decoder->sil))
    {
      float lmscoreend;
      BMRLMState *newlmstate = BMRLM_finish(decoder->lm, lmstate, &lmscoreend);
      BMRDecoder_candidates_add(decoder, opt, newlmstate, lex, prevhyp, prevhyp->score+opt->lmweight*lmscoreend, 0, NULL);
    }
  }
  BMRDecoder_candidates_store(decoder, opt, BMRArray_get(hyp, T+1), 1); /* sort */

  if(nhyp_) {
    BMRDecoder_store_hypothesis(decoder, nhyp_, scores_, llabels_, labels_);
  }
}

void BMRDecoder_store_hypothesis(BMRDecoder *decoder, long *nhyp_, float *scores_, long *llabels_, long *labels_)
{
  long T = BMRArray_size(decoder->hyp)-3;
  for(long r = 0; r < BMRArray_size(BMRArray_get(decoder->hyp, T+1)); r++) {
    BMRDecoderNode *node = BMRArray_get(BMRArray_get(decoder->hyp, T+1), r);
    if(scores_) {
      scores_[r] = node->score;
    }
    long i = 0;
    while(node) {
      if(labels_) {
        labels_[r*(T+2)+T+1-i] = (node->label ? node->label->usr : -1);
      }
      if(llabels_) {
        llabels_[r*(T+2)+T+1-i] = BMRTrieNode_idx(node->lex);
      }
      node = node->parent;
      i++;
    }
  }
  if(nhyp_) {
    *nhyp_ = BMRArray_size(BMRArray_get(decoder->hyp, T+1));
  }
}

double BMRDecoder_forward(BMRDecoder *decoder, BMRDecoderOptions *opt, float *transitions, float *emissions, long T, long N)
{
  BMRDecoder_decode(decoder, opt, transitions, emissions, T, N, NULL, NULL, NULL, NULL);
  BMRDecoderNodeList *merged = NULL;
  BMRArray *lhyp = BMRArray_get(decoder->hyp, T+1);
  for(long i = 1; i < BMRArray_size(lhyp); i++) { // keep best for finalize
    BMRDecoderNode *node = BMRArray_get(lhyp, i);
    BMRDecoderNode *nodeclone = BMRDecoder_node_new(decoder->candidates, node->lmstate, node->lex, node, node->score, node->wordscore, NULL, NULL);
    merged = BMRDecoder_node_list_new(decoder->merged, nodeclone, merged);
  }
  // DEBUG: for now we finalize with logadd (forced)
  // ... so at least we consider all the final paths in the beam
  int logadd = opt->logadd;
  opt->logadd = 1;
  BMRDecoderNode *node = BMRArray_get(lhyp, 0);
  BMRDecoderNode *nodeclone = BMRDecoder_node_new(decoder->candidates, node->lmstate, node->lex, node, node->score, node->wordscore, NULL, NULL);
  BMRDecoderNode *mergedscorenode = BMRDecoder_finalize_merge(decoder, opt, nodeclone, merged);
  BMRArray_set(BMRArray_get(decoder->hyp, T+2), 0, mergedscorenode);
  opt->logadd = logadd;
  return mergedscorenode->score;
}

void BMRDecoder_backward(BMRDecoder *decoder, BMRDecoderOptions *opt, float *gtransitions, float *gemissions, long T, long N, double g)
{
  // init the beast
  BMRDecoderNode *mergedscorenode = BMRArray_get(BMRArray_get(decoder->hyp, T+2), 0);
  mergedscorenode->gscore = g;
  mergedscorenode->active = 1;

  for(long t = T+1; t >= 0; t--) {
    // t+1 slightly confusing, but corresponds to forward loop
    BMRArray *hyp_t = BMRArray_get(decoder->hyp, t+1);
    for(long i = 0; i < BMRArray_size(hyp_t); i++) {
      BMRDecoderNode *node = BMRArray_get(hyp_t, i);
      if(node->active) {
        if(node->merged) {
          if(node == mergedscorenode) {
            BMRDecoderNodeList_dlogadd(node->merged, node->gscore); // DEBUG: for now we finalize with logadd (forced)
          } else {
            if(opt->logadd) {
              BMRDecoderNodeList_dlogadd(node->merged, node->gscore);
            } else {
              BMRDecoderNodeList_dmax(node->merged, node->gscore);
            }
          }
          BMRDecoderNodeList *lst = node->merged;
          while(lst) {
            lst->node->active = 1;
            if(lst->node->parent) {
              lst->node->parent->active = 1;
              lst->node->parent->gscore += lst->node->gscore;
              if(t > 0 && t < T) { // lst->node->parent exists
                long n = BMRTrieNode_idx(lst->node->lex);
                long prevn = BMRTrieNode_idx(lst->node->parent->lex);
                gtransitions[n*N+prevn] += lst->node->gscore;
              }
            }
            if(t >= 0 && t < T) {
              long n = BMRTrieNode_idx(lst->node->lex);
              gemissions[t*N+n] += lst->node->gscore;
            }
            lst = lst->next;
          }
        } else { // not merged
          if(node->parent) { // DEBUG: should exists, no? (t > 0)
            node->parent->active = 1;
            node->parent->gscore += node->gscore;
            if(t > 0 && t < T) { // node->parent exists
              long n = BMRTrieNode_idx(node->lex);
              long prevn = BMRTrieNode_idx(node->parent->lex);
              gtransitions[n*N+prevn] += node->gscore;
            }
          }
          if(t >= 0 && t < T) {
            long n = BMRTrieNode_idx(node->lex);
            gemissions[t*N+n] += node->gscore;
          }
        }
      }
    }
  }
}

static BMRDecoderNodeList* BMRDecoder_haspath_addcheck(BMRDecoderNodeList *matchlst, BMRBuffer *buffer, BMRDecoderNode *node, long label)
{
  if(!node) {
    printf("Decoder: fatal error: node is null (internal bug, please report)\n");
    exit(EXIT_FAILURE);
  }
  if(node->merged) {
    BMRDecoderNodeList *lst = node->merged;
    while(lst) {
      if(BMRTrieNode_idx(lst->node->lex) == label) {
        BMRDecoderNodeList *match = BMRBuffer_grow(buffer, BMRBuffer_size(buffer)+1);
        match->node = lst->node->parent;
        match->next = matchlst;
        matchlst = match;
      }
      lst = lst->next;
    }
  } else {
    if(BMRTrieNode_idx(node->lex) == label) {
      BMRDecoderNodeList *match = BMRBuffer_grow(buffer, BMRBuffer_size(buffer)+1);
      match->node = node->parent;
      match->next = matchlst;
      matchlst = match;
    }
  }
  return matchlst;
}

int BMRDecoder_haspath(BMRDecoder *decoder, long *path, long T) // size == T+2
{
  BMRArray *lhyp = BMRArray_get(decoder->hyp, T+1);
  BMRDecoderNodeList *matchlst = NULL;
  BMRBuffer *buffer = BMRBuffer_new(sizeof(BMRDecoderNodeList));
  BMRDecoderNodeList *newmatchlst = NULL;
  BMRBuffer *newbuffer = BMRBuffer_new(sizeof(BMRDecoderNodeList));

  for(long i = 0; i < BMRArray_size(lhyp); i++) {
    BMRDecoderNode *node = BMRArray_get(lhyp, i);
    matchlst = BMRDecoder_haspath_addcheck(matchlst, buffer, node, path[T+1]);
  }
  for(long t = T; t >= 0 && matchlst; t--) {
    newmatchlst = NULL;
    BMRBuffer_reset(newbuffer);
    while(matchlst) {
      newmatchlst = BMRDecoder_haspath_addcheck(newmatchlst, newbuffer, matchlst->node, path[t]);
      matchlst = matchlst->next;
    }
    matchlst = newmatchlst;
    BMRBuffer *tmp = buffer;
    buffer = newbuffer;
    newbuffer = tmp;
  }
  BMRBuffer_free(buffer);
  BMRBuffer_free(newbuffer);
  return (matchlst != NULL);
}

long BMRDecoder_mem(BMRDecoder *decoder)
{
  return BMRBuffer_mem(decoder->nodes)+BMRBuffer_mem(decoder->candidates)+BMRBuffer_mem(decoder->merged)+sizeof(BMRDecoder);
}

void BMRDecoder_free(BMRDecoder *decoder)
{
  if(decoder) {
    BMRBuffer_free(decoder->nodes);
    BMRBuffer_free(decoder->candidates);
    BMRBuffer_free(decoder->merged);
    for(long i = 0; i < BMRArray_size(decoder->hyp); i++) {
      BMRArray_free(BMRArray_get(decoder->hyp, i));
    }
    BMRArray_free(decoder->hyp);
    free(decoder);
  }
}

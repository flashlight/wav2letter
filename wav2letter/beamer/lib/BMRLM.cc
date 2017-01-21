/* (c) Ronan Collobert 2016, Facebook */

#include "lm/model.hh"

extern "C" {
#include "BMRBuffer.h"
}

typedef struct BMRLM_ {
  BMRBuffer *states;
  lm::base::Model *model;
  const lm::base::Vocabulary *vocab;
} BMRLM;

typedef void BMRLMState;


extern "C" {

BMRLM* BMRLM_new(const char *path)
{
  lm::base::Model *model;

  try {
    model = lm::ngram::LoadVirtual(path);
  } catch(const std::exception &e) {
    return NULL;
  }

  BMRLM *lm = (BMRLM*)malloc(sizeof(BMRLM));
  if(lm) {
    lm->model = model;
    lm->vocab = &model->BaseVocabulary();
    lm->states = BMRBuffer_new(model->StateSize());
    if(!lm->states) {
      delete model;
      free(lm);
      return NULL;
    }
  }

  return lm;
}

long BMRLM_index(BMRLM *lm, const char *word)
{
  return lm->vocab->Index(word);
}

BMRLMState* BMRLM_start(BMRLM *lm, int isnull)
{
  BMRBuffer_reset(lm->states);
  BMRLMState *state = BMRBuffer_grow(lm->states, BMRBuffer_size(lm->states)+1);
  if(isnull) {
    lm->model->NullContextWrite(state);
  }
  else {
    lm->model->BeginSentenceWrite(state);
  }
  return state;
}

BMRLMState* BMRLM_score(BMRLM *lm, BMRLMState *inState, long wordidx, float *score_)
{
  BMRLMState *outState = BMRBuffer_grow(lm->states, BMRBuffer_size(lm->states)+1);
  *score_ = lm->model->BaseScore(inState, wordidx, outState);
  return outState;
}

BMRLMState* BMRLM_scorebest(BMRLM *lm, BMRLMState *inState, long* words, long nwords, float *score_, long* nextword_)
{
  long i;
  BMRLMState *outState = BMRBuffer_grow(lm->states, BMRBuffer_size(lm->states)+1);
  long bestid = 0;
  float bestsc = lm->model->BaseScore(inState, words[0], outState);

  for (i=1; i<nwords; i++){
      float sc = lm->model->BaseScore(inState, words[i], outState);
      if (sc > bestsc){
          bestsc = sc;
          bestid = i;
      }
  }
  *score_ = lm->model->BaseScore(inState, words[bestid], outState);
  *nextword_ = bestid;
  return outState;
}

void BMRLM_scoreall(BMRLM *lm, BMRLMState *inState, long* words, long nwords, float *score_)
{
  long i;
  BMRLMState *outState = BMRBuffer_grow(lm->states, BMRBuffer_size(lm->states)+1);
  for (i=0; i<nwords; i++){
      score_[i] = lm->model->BaseScore(inState, words[i], outState);
  }
}

BMRLMState* BMRLM_finish(BMRLM *lm, BMRLMState *inState, float *score_)
{
  /* DEBUG: could skip the end sentence </s> */
  BMRLMState *outState = BMRBuffer_grow(lm->states, BMRBuffer_size(lm->states)+1);
  *score_ = lm->model->BaseScore(inState, lm->vocab->EndSentence(), outState);
  return outState;
}

void BMRLM_free(BMRLM *lm)
{
  if(lm) {
    delete lm->model;
    BMRBuffer_free(lm->states);
    free(lm);
  }
}

long BMRLM_mem(BMRLM *lm)
{
  return BMRBuffer_mem(lm->states)+sizeof(BMRLM);
}

int BMRLMState_compare(BMRLMState *state1_, BMRLMState *state2_)
{
  lm::ngram::State *state1 = (lm::ngram::State*)state1_;
  lm::ngram::State *state2 = (lm::ngram::State*)state2_;
  return state1->Compare(*state2);
}

}
